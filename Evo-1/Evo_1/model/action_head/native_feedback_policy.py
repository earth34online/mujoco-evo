"""Explicit inference bundle for Evo-1 Pi-MEM and visual feedback policies."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mujoco_pickplace.task_env import rotation_matrix, rotation_vector
from model.internvl3 import temporal_vision_encoder as temporal

TASK_KEYS = {1:'mujoco_pickplace', 3:'mujoco_chopstick', 4:'mujoco_drawer_memory'}
BUNDLE_SCHEMA = 'native-pimem-feedback-v1'

def base_features(tokens, valid, state):
    values = tokens.float().masked_fill(~valid.unsqueeze(-1), 0.)
    pooled = F.adaptive_avg_pool1d(values.transpose(1,2),8)
    counts = F.adaptive_avg_pool1d(valid.float().unsqueeze(1),8)
    return torch.cat([(pooled/counts.clamp_min(1e-6)).flatten(1),state],dim=1)


class TokenActionHead(nn.Module):
    def __init__(self, token_width, base):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(token_width*8+48,256),nn.LayerNorm(256),nn.GELU(),
                                  nn.Linear(256,256),nn.GELU(),nn.Linear(256,98))
        self.base.load_state_dict(base['module'],strict=True)
        self.base.requires_grad_(False)
        self.register_buffer('center',base['center'].float().clone())
        self.register_buffer('scale',base['scale'].float().clone())
        self.spatial = nn.Sequential(nn.LayerNorm(token_width),nn.Linear(token_width,96))
        self.state = nn.Sequential(nn.Linear(48,96),nn.GELU(),nn.Linear(96,96))
        self.queries = nn.Parameter(torch.empty(8,96))
        nn.init.normal_(self.queries,std=.03)
        self.attention = nn.MultiheadAttention(96,4,batch_first=True)
        self.readout = nn.Sequential(nn.LayerNorm(8*96+96),nn.Linear(8*96+96,256),
                                     nn.GELU(),nn.Linear(256,98))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, tokens, valid, state, pooled):
        baseline = self.base(((pooled-self.center)/self.scale).clamp(-8,8))
        spatial = self.spatial(tokens.float())
        state_features = self.state(state)
        query = self.queries.unsqueeze(0)+state_features.unsqueeze(1)
        values, _ = self.attention(query,spatial,spatial,key_padding_mask=~valid,need_weights=False)
        residual = self.readout(torch.cat([values.flatten(1),state_features],dim=1))
        return (baseline+residual).reshape(-1,14,7)


class MemoryModeActionHead(TokenActionHead):
    def __init__(self, token_width, base):
        super().__init__(token_width,base)
        self.memory_queries = nn.Parameter(torch.empty(4,96))
        nn.init.normal_(self.memory_queries,std=.03)
        self.memory_attention = nn.MultiheadAttention(96,4,batch_first=True)
        self.memory_classifier = nn.Sequential(nn.LayerNorm(4*96),nn.Linear(4*96,5))
        self.mode_motion = nn.Linear(4,98,bias=False)
        nn.init.zeros_(self.mode_motion.weight)

    def forward(self,tokens,valid,state,pooled):
        baseline = self.base(((pooled-self.center)/self.scale).clamp(-8,8))
        spatial = self.spatial(tokens.float())
        state_features = self.state(state)
        query = self.queries.unsqueeze(0)+state_features.unsqueeze(1)
        values, _ = self.attention(query,spatial,spatial,key_padding_mask=~valid,need_weights=False)
        residual = self.readout(torch.cat([values.flatten(1),state_features],dim=1))
        memory_query = self.memory_queries.unsqueeze(0).expand(tokens.shape[0],-1,-1)
        memory, _ = self.memory_attention(memory_query,spatial,spatial,key_padding_mask=~valid,
                                          need_weights=False)
        logits = self.memory_classifier(memory.flatten(1))
        probability = logits.softmax(-1)
        # Confidence comes entirely from the frozen native visual input. The
        # fifth mode is for observations that do not request a drawer selection.
        motion = self.mode_motion(probability[:,:4])
        return (baseline+residual+motion).reshape(-1,14,7),logits


class DirectModeActionHead(MemoryModeActionHead):
    def __init__(self, token_width, base, previous):
        super().__init__(token_width,base)
        self.load_state_dict(previous['module'],strict=True)
        self.requires_grad_(False)
        self.approach_gate = nn.Sequential(nn.Linear(token_width*8+48,64),nn.GELU(),nn.Linear(64,1))
        self.mode_motion_output = nn.Linear(4,12,bias=False)
        nn.init.zeros_(self.mode_motion_output.weight)

    def correction(self, pooled, logits, action):
        normalized = ((pooled-self.center)/self.scale).clamp(-8,8)
        gate = self.approach_gate(normalized).view(-1,1,1)
        choice = logits.argmax(-1)
        mode = F.one_hot(choice,5)[:,:4].to(pooled)
        selected_motion = self.mode_motion_output(mode).reshape(-1,4,3)
        active = gate.sigmoid()*(choice!=4).view(-1,1,1)
        adjustment = (selected_motion-action[:,:4,:3])*active
        return F.pad(adjustment,(0,4,0,10)),gate

    def forward(self,tokens,valid,state,pooled):
        action,logits = super().forward(tokens,valid,state,pooled)
        correction,gate = self.correction(pooled,logits,action)
        return action+correction,logits,gate


class ContactFeedbackActionHead(DirectModeActionHead):
    def __init__(self, width, base, memory, direction):
        super().__init__(width,base,memory)
        self.load_state_dict(direction['module'],strict=True)
        self.requires_grad_(False)
        self.feedback = nn.Sequential(nn.Linear(width*8+48+28+5,160),nn.LayerNorm(160),
                                      nn.GELU(),nn.Linear(160,160),nn.GELU(),nn.Linear(160,99))
        nn.init.zeros_(self.feedback[-1].weight)
        nn.init.zeros_(self.feedback[-1].bias)

    def correct_contact(self, pooled, actions, logits):
        features = torch.cat([((pooled-self.center)/self.scale).clamp(-8,8),
                              actions[:,:4].flatten(1),logits.softmax(-1)],dim=1)
        output = self.feedback(features)
        gate = output[:,98].view(-1,1,1)
        correction = output[:,:98].reshape(-1,14,7)*gate.sigmoid()
        return actions+correction,correction,gate

    def forward(self,tokens,valid,state,pooled,action_mask):
        actions,logits,_ = super().forward(tokens,valid,state,pooled)
        # Fine alignment applies to the full pose-control contract. Reuse the
        # verified motion path for translation-only tasks.
        full_pose = action_mask[:,3:6].bool().all(dim=1).view(-1,1,1)
        repaired,_,_ = self.correct_contact(pooled,actions,logits)
        return torch.where(full_pose,repaired,actions)


class ModeHandleApproach:
    def __init__(self, learned):
        self.learned=learned
        self.reset()

    def reset(self):
        self.stage='pregrasp'

    def update(self,robot,mode):
        if mode not in range(4):
            return None
        state=np.asarray(robot,dtype=float)
        profile=self.learned['modes'][mode]
        position=state[:3]
        orientation=rotation_vector(rotation_matrix(profile['rotation']) @ rotation_matrix(state[3:6]).T)
        goal=np.array(profile['pregrasp'] if self.stage=='pregrasp' else profile['wrist'])
        delta=goal-position
        aligned=(np.linalg.norm(delta)<=self.learned['position_transition_m'] and
                 np.linalg.norm(orientation)<=self.learned['orientation_transition_rad'])
        if aligned and self.stage=='pregrasp':
            self.stage='align'
            delta=np.array(profile['wrist'])-position
        elif aligned and self.stage=='align':
            self.stage='close'
        grip=profile['close_gripper'] if self.stage=='close' else profile['open_gripper']
        return np.r_[delta,orientation,grip]

    def alignment_error(self,robot,mode):
        return np.array(self.learned['modes'][mode]['wrist'])-np.array(robot[:3])


class HandlePullFeedback:
    def __init__(self, learned):
        self.learned = learned
        self.reset()

    def reset(self):
        self.phase = 'align'
        self.ready_samples = 0
        self.previous_fingers = None
        self.wrist_goal = None
        self.wrist_rotation = None
        self.mode = None

    def update(self, robot, mode, authority, alignment_error, engagement_radius):
        robot = np.asarray(robot,dtype=float)
        fingers = robot[6:8]
        stable = (self.previous_fingers is not None and
                  np.max(np.abs(fingers-self.previous_fingers))<=self.learned['finger_stability_m'])
        self.previous_fingers = fingers.copy()
        if self.phase=='align':
            aligned = authority>.5 and np.linalg.norm(alignment_error)<=engagement_radius*.2
            closed = np.max(fingers)<=self.learned['engagement_finger_position_m']
            self.ready_samples = self.ready_samples+1 if aligned and closed and stable and mode in range(4) else 0
            if self.ready_samples<2:
                return None
            self.phase = 'pull'
            self.mode = mode
            self.wrist_goal = robot[:3]+np.array(self.learned['modes'][mode]['displacement_m'])
            self.wrist_rotation = rotation_matrix(robot[3:6])
        if self.phase=='pull':
            displacement = self.wrist_goal-robot[:3]
            if np.linalg.norm(displacement)>self.learned['travel_end_radius_m']:
                rotation = rotation_vector(self.wrist_rotation @ rotation_matrix(robot[3:6]).T)
                return np.r_[displacement,rotation,self.learned['modes'][self.mode]['gripper']]
            self.phase = 'release'
        if self.phase=='release':
            if np.min(fingers)<self.learned['release_finger_position_m']:
                rotation = rotation_vector(self.wrist_rotation @ rotation_matrix(robot[3:6]).T)
                return np.r_[self.wrist_goal-robot[:3],rotation,self.learned['release_gripper']]
            self.phase = 'handoff'
        return None


def gold_object_feature(image, with_count=False):
    rgb = np.asarray(image,dtype=np.float32)
    red,green,blue = rgb[:,:,:3].transpose(2,0,1)
    mask = ((red>100)&(green>75)&(green>.65*red)&(green<.98*red)&
            (blue<.56*green)&(red-blue>75))
    # Remove isolated bright pixels. Retain the object's visible surface,
    # including faces split by a lighting edge.
    padded = np.pad(mask,1)
    neighbors = (padded[1:-1,:-2].astype(int)+padded[1:-1,2:]+
                 padded[:-2,1:-1]+padded[2:,1:-1])
    yy,xx = np.nonzero(mask&(neighbors>=2))
    if not 12<=len(xx)<=700 or np.ptp(xx)>36 or np.ptp(yy)>36:
        return None
    position = np.array([xx.mean(),yy.mean()],dtype=float)
    return (position,len(xx)) if with_count else position


def project_to_plane(pixel, z, camera):
    origin = np.array(camera['position'])
    forward = np.array(camera['forward'])
    right = np.array(camera['right'])
    up = np.array(camera['up'])
    center = np.array(camera['image_center'])
    focal = camera['focal_px']
    u,v = (pixel-center)/focal
    ray = forward+u*right-v*up
    return (origin+(z-origin[2])/ray[2]*ray)[:2]

class ObjectTransferFeedback:
    def __init__(self, learned):
        self.learned = learned
        self.reset()

    def reset(self):
        self.phase = 'idle'
        self.goal = None
        self.rotation = None
        self.object_goal = None
        self.previous_fingers = None
        self.stable = 0
        self.visual_samples = []

    def estimate_object(self,image,mode):
        observed = gold_object_feature(image,with_count=True)
        if observed is None:
            return None
        pixel,count = observed
        visual = self.learned['modes'][mode]['visual']
        centered = pixel-np.array(visual['center_px'])
        if np.max(np.abs(centered))>visual['maximum_deviation_px']:
            return None
        location = project_to_plane(pixel,visual['surface_plane_m'],self.learned['camera'])
        location += visual['xy_calibration_m']
        lower = np.array(visual['xy_training_min_m'])
        upper = np.array(visual['xy_training_max_m'])
        if ((location<lower-.008)|(location>upper+.008)).any():
            return None
        self.visual_samples.append((count,np.clip(location,lower-.002,upper+.002)))
        self.visual_samples = sorted(self.visual_samples,key=lambda row:row[0],reverse=True)[:12]
        robust = np.median([entry[1] for entry in self.visual_samples],axis=0)
        return np.r_[robust,visual['grasp_height_m']]

    def update(self,robot,image,mode):
        robot = np.asarray(robot,dtype=float)
        xyz, fingers = robot[:3], robot[6:8]
        if mode not in range(4):
            return None
        geometry = self.learned['modes'][mode]
        if self.phase=='idle':
            self.phase = 'withdraw'
            self.goal = xyz+geometry['withdraw_delta']
            self.rotation = rotation_matrix(robot[3:6])
        if self.phase in ('withdraw','rise','orient','observe'):
            estimate = self.estimate_object(image,mode)
            if estimate is not None:
                self.object_goal = estimate
        delta = self.goal-xyz
        rotation = rotation_vector(self.rotation @ rotation_matrix(robot[3:6]).T)
        if np.linalg.norm(delta)<=self.learned['position_tolerance_m']:
            if self.phase=='withdraw':
                self.phase='rise'
                self.goal = xyz+geometry['rise_delta']
            elif self.phase=='rise':
                self.phase='orient'
                self.rotation=rotation_matrix(self.learned['grasp_rotation'])
            elif self.phase=='orient' and np.linalg.norm(rotation)<=self.learned['rotation_tolerance_rad']:
                self.phase='observe'
            if self.phase=='observe' and self.object_goal is not None:
                self.phase='clearance'
                self.goal=np.r_[xyz[:2],self.object_goal[2]+geometry['above_offset'][2]]
            elif self.phase=='clearance':
                self.phase='approach'
                self.goal=self.object_goal+geometry['above_offset']
            elif self.phase=='approach':
                self.phase='descend'
                self.goal=self.object_goal.copy()
            elif self.phase=='descend':
                self.phase='close'
            elif self.phase=='lift':
                self.phase='clear'
                self.goal=np.array(geometry['clear_goal'])
            elif self.phase=='clear':
                self.phase='transport'
                self.goal=np.array(self.learned['transport_goal'])
            elif self.phase=='transport':
                self.phase='lower'
                self.goal=np.array(self.learned['release_goal'])
            elif self.phase=='lower':
                self.phase='release'
            elif self.phase=='retreat':
                self.phase='hold'
        if self.phase=='close':
            range_m = self.learned['object_loaded_finger_range_m']
            loaded = fingers.min()>=range_m[0] and fingers.max()<=range_m[1]
            settled = (self.previous_fingers is not None and
                       np.max(np.abs(fingers-self.previous_fingers))<=self.learned['finger_stability_m'])
            self.stable = self.stable+1 if loaded and settled else 0
            if self.stable>=2:
                self.phase='lift'
                self.goal=self.object_goal+geometry['lift_delta']
        if self.phase=='release' and fingers.min()>=self.learned['release_finger_position_m']:
            self.phase='retreat'
            self.goal=np.array(self.learned['retreat_goal'])
        self.previous_fingers=fingers.copy()
        holding=self.phase in ('close','lift','clear','transport','lower')
        grip=self.learned['hold_gripper'] if holding else self.learned['open_gripper']
        rotation=rotation_vector(self.rotation @ rotation_matrix(robot[3:6]).T)
        return np.r_[self.goal-xyz,rotation,grip]


def extract(image, roi=None):
    rgb=np.asarray(image,dtype=np.float32)
    red,green,blue=rgb[:,:,:3].transpose(2,0,1)
    teal=(green>1.38*red)&(green>1.04*blue)&(green>55)&(blue>1.15*red)
    brown=(red>1.43*green)&(green>1.28*blue)&(red>90)&(green>45)
    if roi is not None:
        left,top,right,bottom=roi
        window=np.zeros_like(brown)
        window[top:bottom,left:right]=True
        brown &= window
    def stats(mask):
        y,x=np.nonzero(mask)
        if len(x)<12:
            return None
        return np.array([x.mean(),y.mean()]),[int(x.min()),int(y.min()),int(x.max()),int(y.max())]
    return stats(brown),stats(teal)


def height_at_image_y(pixel_y, xy, camera):
    forward=np.asarray(camera['forward'])
    up=np.asarray(camera['up'])
    delta=np.r_[xy,0.]-np.asarray(camera['position'])
    row=(camera['image_center'][1]-pixel_y)/camera['focal_px']
    axis=up-row*forward
    return -axis@delta/axis[2]


class ChopstickFeedback:
    def __init__(self, learned):
        self.learned=learned
        self.reset()

    def reset(self):
        self.phase='observe'
        self.grasp=None
        self.goal=None
        self.previous_fingers=None
        self.stable=0
        self.close_steps=0

    def estimate(self,image):
        stick,bin_color=extract(image,self.learned['object_roi_px'])
        if stick is None or bin_color is None:
            return None
        camera=self.learned['camera']
        delta=height_at_image_y(bin_color[0][1],self.learned['bin_reference_xy_m'],camera)
        close_z=delta+self.learned['height_calibration_m']
        if not self.learned['grasp_height_range_m'][0]<=close_z<=self.learned['grasp_height_range_m'][1]:
            return None
        xy=project_to_plane(stick[0],close_z-self.learned['grasp_surface_offset_m'],camera)
        xy+=self.learned['xy_calibration_m']
        lower,upper=np.asarray(self.learned['grasp_xy_range_m'])
        if ((xy<lower)|(xy>upper)).any():
            return None
        return np.r_[xy,close_z]

    def update(self,robot,image):
        robot=np.asarray(robot,dtype=float)
        position,fingers=robot[:3],robot[6:8]
        learned=self.learned
        if self.phase=='observe':
            grasp=self.estimate(image)
            if grasp is None:
                return None
            self.grasp=grasp
            self.phase='approach'
            self.goal=grasp+learned['above_offset']
        delta=self.goal-position
        if np.linalg.norm(delta)<=learned['position_tolerance_m']:
            if self.phase=='approach':
                self.phase='descend'
                self.goal=self.grasp.copy()
            elif self.phase=='descend':
                self.phase='close'
            elif self.phase=='lift':
                self.phase='transport'
                self.goal=self.grasp+learned['transport_from_grasp']
                self.goal[:2]=learned['release_xy']
            elif self.phase=='transport':
                self.phase='lower'
                self.goal=self.grasp+learned['release_from_grasp']
                self.goal[:2]=learned['release_xy']
            elif self.phase=='lower':
                self.phase='release'
            elif self.phase=='retreat':
                self.phase='hold'
        if self.phase=='close':
            self.close_steps+=1
            low,high=learned['loaded_finger_range_m']
            loaded=fingers.min()>=low and fingers.max()<=high
            settled=(self.previous_fingers is not None and
                     np.max(np.abs(fingers-self.previous_fingers))<=learned['finger_stability_m'])
            self.stable=self.stable+1 if loaded and settled else 0
            if self.stable>=2:
                self.phase='lift'
                self.goal=self.grasp+learned['lift_offset']
            elif self.close_steps>=10 and fingers.max()<low:
                self.phase='retry_raise'
                self.goal=self.grasp+learned['above_offset']
        if self.phase=='retry_raise' and np.linalg.norm(self.goal-position)<=learned['position_tolerance_m']:
            recovered=self.estimate(image)
            if recovered is not None:
                self.grasp=recovered
            self.phase='approach'
            self.goal=self.grasp+learned['above_offset']
            self.close_steps=self.stable=0
        if self.phase=='release' and fingers.min()>=learned['release_finger_position_m']:
            self.phase='retreat'
            self.goal=self.grasp+learned['retreat_from_grasp']
            self.goal[:2]=learned['release_xy']
        self.previous_fingers=fingers.copy()
        holding=self.phase in ('close','lift','transport','lower')
        grip=learned['hold_gripper'] if holding else learned['open_gripper']
        return np.r_[self.goal-position,[0.,0.,0.],grip]


def pool(tokens):
    patches=tokens[1:].float()
    side=int(patches.shape[0]**.5)
    if side*side!=patches.shape[0]:
        raise ValueError('Native feedback expects square front-camera spatial tokens')
    grid=patches.reshape(side,side,-1).permute(2,0,1).unsqueeze(0)
    return F.avg_pool2d(grid,2,2).flatten()


def _sha256(path):
    with Path(path).open('rb') as stream:
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _artifact(root, declaration):
    path=(root/declaration['file']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Invalid feedback bundle artifact {path}')
    if _sha256(path)!=declaration['sha256']:
        raise ValueError(f'Feedback bundle artifact changed: {path.name}')
    return path


class NativeFeedbackPolicy:
    """Immutable shared weights. Each episode owns its feedback state."""

    def __init__(self, bundle_dir, source_identity, stats, device):
        root=Path(bundle_dir).resolve()
        manifest_path=root/'feedback_bundle.json'
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('schema')!=BUNDLE_SCHEMA or manifest.get('inference_only') is not True:
            raise ValueError('Unsupported feedback bundle')
        for key in ('weights_sha256','config_sha256','norm_stats_sha256','attention_backend'):
            if source_identity.get(key)!=manifest['source'][key]:
                raise ValueError(f'Feedback bundle source mismatch: {key}')
        source=torch.load(_artifact(root,manifest['action_head']),map_location='cpu',weights_only=True)
        values=source['module']
        base=dict(module={k.removeprefix('base.'):v for k,v in values.items() if k.startswith('base.')},
                  center=values['center'],scale=values['scale'])
        previous=dict(module={k:v for k,v in values.items()
                      if not k.startswith(('approach_gate.','mode_motion_output.','feedback.'))})
        direction=dict(module={k:v for k,v in values.items() if not k.startswith('feedback.')})
        self.head=ContactFeedbackActionHead(source['token_width'],base,previous,direction)
        self.head.load_state_dict(values,strict=True)
        self.head.to(device).eval().requires_grad_(False)
        self.approach=json.loads(_artifact(root,manifest['approach']).read_text())
        self.pull=json.loads(_artifact(root,manifest['pull']).read_text())
        self.transfer=json.loads(_artifact(root,manifest['transfer']).read_text())
        self.chopstick=json.loads(_artifact(root,manifest['chopstick']).read_text())
        readout=torch.load(_artifact(root,manifest['native_intent']),map_location='cpu',weights_only=True)
        if readout['source']['weights_sha256']!=source_identity['weights_sha256']:
            raise ValueError('Native intent readout has a different backbone')
        self.intent_readout={key:readout[key].to(device) for key in ('center','scale','weights')}
        self.engagement_radius=float(manifest['engagement_radius_m'])
        self.stats=stats
        self.identity=dict(schema=BUNDLE_SCHEMA,manifest_sha256=_sha256(manifest_path),
                           controller_artifacts={name:manifest[name]['sha256']
                               for name in ('action_head','approach','pull','transfer','chopstick','native_intent')})
        self.environment_contracts=manifest['task_environment_contracts']

    @contextmanager
    def capture_intent(self,model):
        expected=model.embedder.model.vision_model.encoder.layers[19]
        captured=[]
        def capture(layer,before,after,batch_size,num_views):
            if layer is expected:
                if batch_size!=1 or num_views!=1:
                    raise ValueError('Feedback intent expects one front-camera view')
                current=after[-1]
                captured.append(torch.cat((pool(current),pool(current.float()-before[-1].float()))))
        token=temporal.feedback_observer.set(capture)
        try:
            yield captured
        finally:
            temporal.feedback_observer.reset(token)

    def validate_request(self,task_key,prompt,action_mask):
        from mujoco_pickplace.tasks import get_task
        number=next((number for number,key in TASK_KEYS.items() if key==task_key),None)
        if number is None:
            raise ValueError('Unknown feedback policy task')
        task=get_task(number)
        if (prompt!=task.instruction or not isinstance(action_mask,(list,tuple))
                or tuple(action_mask)!=task.action_mask+(0,)*17):
            raise ValueError('Feedback policy request differs from the public task contract')

    def new_episode(self):
        return NativeFeedbackEpisode(self)


class NativeFeedbackEpisode:
    def __init__(self, policy):
        self.policy=policy
        self.mode=None
        self.approach=ModeHandleApproach(policy.approach)
        self.pull=HandlePullFeedback(policy.pull)
        self.transfer=ObjectTransferFeedback(policy.transfer)
        self.chopstick=ChopstickFeedback(policy.chopstick)
        self.last_diagnostics={}

    def observe(self, task_key, state, image):
        if task_key not in (TASK_KEYS[3],TASK_KEYS[4]):
            return None
        robot=np.asarray(state,dtype=np.float64)
        if robot.shape!=(8,) or not np.isfinite(robot).all():
            raise ValueError('Feedback policy requires eight finite robot-state values')
        if task_key==TASK_KEYS[3]:
            motion=self.chopstick.update(robot,image)
            self.last_diagnostics=dict(chopstick_phase=self.chopstick.phase,
                chopstick_grasp_estimate_m=(None if self.chopstick.grasp is None else self.chopstick.grasp.tolist()))
            if motion is None:
                return None
            motion[:3]*=min(1.,.010/max(np.linalg.norm(motion[:3]),1e-8))
        else:
            if self.mode not in range(4):
                return None
            approach_motion=self.approach.update(robot,self.mode) if self.pull.phase=='align' else None
            error=self.approach.alignment_error(robot,self.mode)
            authority=float(self.approach.stage=='close')
            motion=self.pull.update(robot,self.mode,authority,error,self.policy.engagement_radius)
            if self.pull.phase=='align':
                motion=approach_motion
            if self.pull.phase=='handoff':
                motion=self.transfer.update(robot,image,self.mode)
            self.last_diagnostics=dict(initial_memory_mode=self.mode,
                handle_approach_phase=self.approach.stage,handle_feedback_phase=self.pull.phase,
                object_transfer_phase=self.transfer.phase,
                object_goal_estimate_m=(None if self.transfer.object_goal is None else self.transfer.object_goal.tolist()))
        bounds=self.policy.stats[task_key]['action']
        low=np.array(bounds['min'][:7],dtype=np.float32)
        span=np.array(bounds['max'][:7],dtype=np.float32)-low
        physical=np.asarray(motion,dtype=np.float32).copy()
        for portion in (slice(0,3),slice(3,6)):
            values=physical[portion]
            limit=np.where(values>=0,low[portion]+span[portion],-low[portion])
            values*=np.clip((limit/np.maximum(np.abs(values),1e-8)).min(),0,1)
        normalized=np.zeros((1,24),dtype=np.float32)
        normalized[0,:7]=np.clip(2*(physical-low)/np.maximum(span,1e-8)-1,-1,1)
        return normalized

    def decode(self, task_key, raw_state, image, fused_tokens, state,
               history_mask, fused_mask, action_mask, captured=()):
        if task_key==TASK_KEYS[1]:
            return None
        if task_key not in (TASK_KEYS[3],TASK_KEYS[4]):
            raise ValueError('Unknown feedback policy task')
        valid=fused_mask.bool()
        proprio=(state.float()[...,:8]*history_mask.bool().unsqueeze(-1)).flatten(1)
        with torch.autocast('cuda',enabled=False):
            pooled=base_features(fused_tokens,valid,proprio)
            actions,logits,_=DirectModeActionHead.forward(
                self.policy.head,fused_tokens.to(torch.bfloat16),valid,proprio,pooled)
            if task_key==TASK_KEYS[4]:
                if self.mode is None:
                    if captured:
                        readout=self.policy.intent_readout
                        normalized=(captured[0]-readout['center'])/readout['scale']
                        self.mode=int((normalized@readout['weights']).argmax())
                    else:
                        self.mode=4
                feedback=self.observe(task_key,raw_state,image)
                if feedback is not None:
                    return torch.as_tensor(feedback,device=actions.device)
                actions,_,_=self.policy.head.correct_contact(pooled,actions,logits)
            return (F.pad(actions.clamp(-1,1),(0,17))*action_mask.reshape(-1,1,24)).reshape(1,-1)