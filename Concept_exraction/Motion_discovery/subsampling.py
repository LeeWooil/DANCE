import json
import numpy as np
import glob
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import Concept_exraction.utils.motion_discovery_utils as util

def get_keypoints(json_file, confidence_threshold=0.1):
    """JSON 파일에서 keypoints를 추출하고 정규화."""
    with open(json_file, "r") as f:
        keypoints_data = json.load(f).get("keypoints", [])
    no_keyponit_index = []
    frames_data = []
    for i, frame in enumerate(keypoints_data):
        if "0" in frame:
            keypoints = np.array(frame["0"])  # (17, 3)
            mean_confidence = np.mean(keypoints[:, 2])
            if mean_confidence < confidence_threshold:
                continue
            # pose = keypoints[:, :2]
            frames_data.append(keypoints)
        else:
            no_keyponit_index.append(i)
    return frames_data,no_keyponit_index, len(keypoints_data)

def process_keypoints(json_file, scaler, confidence_threshold=0.1):
    """JSON 파일에서 keypoints를 추출하고 정규화."""
    with open(json_file, "r") as f:
        keypoints_data = json.load(f).get("keypoints", [])
    
    frames_data = []
    for frame in keypoints_data:
        if "0" in frame:
            keypoints = np.array(frame["0"])  # (17, 3)
            mean_confidence = np.mean(keypoints[:, 2])
            if mean_confidence < confidence_threshold:
                continue
            pose = keypoints[:, :2]
            mean_pose = np.mean(pose, axis=0)
            pose_centered = pose - mean_pose
            pose_normalized = scaler.fit_transform(pose_centered)
            frames_data.append(pose_normalized)
    return frames_data, mean_confidence

def normalized_keypoints(clip, scaler):
    frames_data = []
    for t in range(clip.shape[0]):  # 각 프레임에 대해
        pose = clip[t]  # shape: (17, 2)
        mean_pose = np.mean(pose, axis=0)  # (2,)
        pose_centered = pose - mean_pose  # (17, 2)
        pose_normalized = scaler.fit_transform(pose_centered)  # (17, 2)
        frames_data.append(pose_normalized)
    return np.array(frames_data)


def subsampling_ver1(args, json_files):
    """샘플을 L등분하고, 각 등분에서 T등분하여 
    랜덤으로 길이 T인 subsequence L개 만들기"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    class_data = []
    class_metadata = []
    
    L, T = args.num_subsequence, args.len_subsequence
    
    for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
        class_name = os.path.basename(os.path.dirname(json_file))
        video_id = os.path.basename(json_file).replace("_result.json", "")
        frames_data = process_keypoints(json_file, scaler)
        
        num_frames = len(frames_data)

        if num_frames == 0:
            print(f"Skipping {json_file} (No valid frames)")
            continue
    
        # 부족한 프레임을 확장
        if num_frames < L * T:
            expanded_frames = util.expand_array(frames_data, L * T)
            if expanded_frames is not None:
                frames_data = expanded_frames
                num_frames = len(frames_data)
            else:
                print(f"Skipping {json_file} (Unable to expand frames)")
                continue

        
        # 평균 구간 길이 계산
        segment_length = num_frames // L
        all_clips = []
        
        for i in range(L):
            segment_start = i * segment_length
            segment_end = (i + 1) * segment_length
            
            # T개의 작은 구간으로 나누기
            sub_segment_length = max(segment_length // T, 1)
            sampled_indices = [segment_start + j * sub_segment_length + np.random.randint(0, sub_segment_length) 
                            for j in range(T)]
            if len(sampled_indices) != T:
                print(len(sampled_indices))
            
            # 샘플링한 인덱스로 클립 생성
            frames_data = np.array(frames_data) 
            clip = frames_data[sampled_indices]
            if len(clip) == T:
                all_clips.append(clip)
                class_metadata.append(video_id)
        
        class_data.extend(all_clips)
    return class_data, class_metadata

def subsampling_ver2(args, json_files):
    """기존 방식. (sliding window 방식) 
    -> 한 칸(stride를 통해 변경 가능)씩 이동하며 샘플링"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    class_data = []
    class_metadata = []
    
    L, T, stride = args.num_subsequence, args.len_subsequence, 1
    
    for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
        class_name = os.path.basename(os.path.dirname(json_file))
        video_id = os.path.basename(json_file).replace("_result.json", "")
        frames_data, missing_index,og_num_frames = get_keypoints(json_file)
        
        num_frames = len(frames_data)

        if num_frames == 0:
            print(f"Skipping {json_file} (No valid frames)")
            continue
        pose = np.array(frames_data)[:, :,:2]
        confidence = np.array(frames_data)[:, :, 2] 
        num_frames = len(pose)
        # 부족한 프레임을 확장
        if num_frames < T:
            pose = util.repeat_to_min_length(pose, T)
            confidence = util.repeat_to_min_length(confidence, T)
        
        for i in range(0, num_frames - T + 1, stride):  # stride만큼 이동동
            clip = np.array(pose[i:i+T])  # (T, 17, 2)
            class_data.append(clip)
            class_metadata.append(video_id)

    return class_data, class_metadata    

def subsampling_ver3(args, json_files):
    """Keyframe을 중심으로 일정 구간을 샘플링"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    class_data = []
    class_metadata = []

    L, T = args.num_subsequence, args.len_subsequence
    
    for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
        video_id = os.path.basename(json_file).replace("_result.json", "")
        frames_data = process_keypoints(json_file, scaler)
        
        num_frames = len(frames_data)
        all_clips = []

        if num_frames == 0:
            print(f"Skipping {json_file} (No valid frames)")
            continue
        
        keyframe_path = os.path.join(args.keyframe_path,video_id, "csvFile", f"{video_id}.txt")
        with open(keyframe_path, 'r') as f:
            keyframes = [int(line.strip()) for line in f.readlines()]
        for keyframe in keyframes:
            # keyframe을 중심으로 앞뒤로 T//2개씩 샘플링
            start = max(keyframe - T//2, 0)  # T개 샘플을 위해 앞뒤로 T//2개씩 선택
            end = min(keyframe + T//2 + 1, num_frames)  # 마지막 프레임을 넘지 않도록

            if end - start < T:
                if start == 0:
                    end = min(start+T, num_frames)
                else :
                    start = max(end-T,0)
            
            sampled_indices = np.arange(start, end)
            
            # 샘플링한 인덱스로 클립 생성
            frames_data = np.array(frames_data)
            clip = frames_data[sampled_indices]
            
            if len(clip) == T:
                all_clips.append(clip)
                class_metadata.append(video_id)
        
        class_data.extend(all_clips)
    
    return class_data, class_metadata

def subsampling_ver4(args, json_files):
    """Keyframe 내에서 uniform sampling"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    class_data = []
    class_metadata = []

    L, T = args.num_subsequence, args.len_subsequence
    
    for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
        video_id = os.path.basename(json_file).replace("_result.json", "")
        frames_data = process_keypoints(json_file, scaler)
        
        num_frames = len(frames_data)
        all_clips = []

        if num_frames == 0:
            print(f"Skipping {json_file} (No valid frames)")
            continue
        
        keyframe_path = os.path.join(args.keyframe_path,video_id, "csvFile", f"{video_id}.txt")
        with open(keyframe_path, 'r') as f:
            keyframes = [int(line.strip()) for line in f.readlines()]
            
        if len(keyframes) == 0 or keyframes[-1] > num_frames:
            print(f"Skipping {keyframe_path} (No valid frames)")
            continue

        if len(keyframes) < L * T:
            expanded_frames = util.expand_array(keyframes, L * T)
            if expanded_frames is not None:
                keyframes_data = expanded_frames
                num_keyframes = len(keyframes_data)
            else:
                print(f"Skipping {json_file} (Unable to expand frames)")
                continue

        
        # 평균 구간 길이 계산
        segment_length = num_keyframes // L
        all_clips = []
        
        for i in range(L):
            segment_start = i * segment_length
            segment_end = (i + 1) * segment_length
            
            # T개의 작은 구간으로 나누기
            sub_segment_length = max(segment_length // T, 1)
            sampled_indices = [segment_start + j * sub_segment_length + np.random.randint(0, sub_segment_length) 
                            for j in range(T)]
            if len(sampled_indices) != T:
                print(len(sampled_indices))
            
            # 샘플링한 인덱스로 클립 생성
            frames_data = np.array(frames_data)
            if sampled_indices[-1] > num_frames:
                continue
            else:
                clip = frames_data[sampled_indices]

            # if np.array(clip).shape != (T,17,2):
            #     print(np.array(clip).shape)

            if len(clip) == T:
                all_clips.append(clip)
                class_metadata.append(video_id)
        
        class_data.extend(all_clips)
    return class_data, class_metadata

def subsampling_ver5(args, json_files):
    """Keyframe을 중심으로 일정 구간을 샘플링
    keyframe 개수에 따라 샘플링 개수 다르게
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    class_data = []
    class_metadata = []

    L, T = args.num_subsequence, args.len_subsequence
    
    for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
        video_id = os.path.basename(json_file).replace("_result.json", "")
        frames_data = process_keypoints(json_file, scaler)
        
        num_frames = len(frames_data)
        all_clips = []

        if num_frames == 0:
            print(f"Skipping {json_file} (No valid frames)")
            continue
        elif num_frames < T:
            frames_data = util.repeat_to_clip(num_frames, T)
            print(f"Original frame len : {num_frames}")
            print(f"Expanding frames to {len(frames_data)}")
        
        # keyframe_path = os.path.join(args.keyframe_path, video_id, "csvFile", f"{video_id}.txt")
        keyframe_path = os.path.join(args.keyframe_path, video_id, f"{video_id}.txt")
        with open(keyframe_path, 'r') as f:
            keyframes = [int(line.strip()) for line in f.readlines()]
            
        if len(keyframes) == 0:
            print(f"Skipping {keyframe_path} (No valid keyframes)")
            continue
        
        # L, keyframe 개수 관계에 따른 처리
        selected_keyframes = []
        
        if L > len(keyframes):
            # L > keyframe: keyframe 개수만큼만
            selected_keyframes = keyframes
        elif L < len(keyframes):
            indices = np.linspace(0, len(keyframes)-1, L, dtype=int)
            selected_keyframes = [keyframes[i] for i in indices]
        else:  # L == len(keyframes)
            selected_keyframes = keyframes
        
        frames_data = np.array(frames_data)
        
        for keyframe in selected_keyframes:
            # keyframe을 중심으로 앞뒤로 T//2개씩 샘플링
            start = max(keyframe - T//2, 0)  # T개 샘플을 위해 앞뒤로 T//2개씩 선택
            end = min(keyframe + T//2 + 1, num_frames)  # 마지막 프레임을 넘지 않도록

            if end - start < T:
                if start == 0:
                    end = min(start+T, num_frames)
                else:
                    start = max(end-T, 0)
            
            sampled_indices = np.arange(start, end)
            
            # 샘플링한 인덱스로 클립 생성
            clip = frames_data[sampled_indices]

            if len(clip) == T:
                all_clips.append(clip)
                class_metadata.append(video_id)
        
        class_data.extend(all_clips)
    
    return class_data, class_metadata
# def subsampling_considering_cos_sim(args, json_files):
#     """Keyframe을 중심으로 일정 구간을 샘플링
#     keyframe 개수에 따라 샘플링 개수 다르게
#     """
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     class_data = []
#     class_metadata = []
#     cnt = 0
#     missing_video =[]
#     L, T = args.num_subsequence, args.len_subsequence
    
#     for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
#         video_id = os.path.basename(json_file).replace("_result.json", "")
#         frames_data, missing_index,og_num_frames = get_keypoints(json_file)
        
        
        
        
#         all_clips = []

#         if len(frames_data) == 0:
#             print(f"Skipping {json_file} (No valid frames)")
#             missing_video.append(json_file)
#             cnt += 1
#             continue
#         pose = np.array(frames_data)[:, :,:2]
#         confidence = np.array(frames_data)[:, :, 2] 
#         num_frames = len(pose)
#         if num_frames < T:
#             pose = util.repeat_to_min_length(pose, T)
#             confidence = util.repeat_to_min_length(confidence, T)
#             print(f"Original frame len : {num_frames}")
#             print(f"Expanding frames to {len(pose)}")
#             num_frames = len(pose)
        
#         keyframe_path = os.path.join(args.keyframe_path, video_id, "csvFile", f"{video_id}.txt")
#         # keyframe_path = os.path.join(args.keyframe_path, f"{video_id}.txt")
#         with open(keyframe_path, 'r') as f:
#             keyframes = [int(line.strip()) for line in f.readlines()]
            
#         # ✅ missing_index 있는 경우
#         if len(missing_index) > 0:
#             # 1. usable 인덱스 만들기
#             valid_indices = [i for i in range(og_num_frames) if i not in missing_index]

#             # 2. keyframes 중 usable 한 것만 남기고, frames_data 기준 인덱스로 재매핑
#             remapped_keyframes = []
#             for kf in keyframes:
#                 if kf in valid_indices:
#                     remapped_keyframes.append(valid_indices.index(kf))  # usable index 기준으로 바꿔줌
#                 else:
#                     print(f"Keyframe {kf} in {video_id} is missing and will be skipped.")

#             keyframes = remapped_keyframes

#             if len(keyframes) == 0:
#                 print(f"Skipping {video_id} (All keyframes are invalid after remapping)")
#                 cnt+=1
#                 continue
            
#         # L, keyframe 개수 관계에 따른 처리
#         selected_keyframes = []
        
#         if L > len(keyframes):
#             # L > keyframe: keyframe 개수만큼만
#             selected_keyframes = keyframes
#         elif L < len(keyframes):
#             indices = np.linspace(0, len(keyframes)-1, L, dtype=int)
#             selected_keyframes = [keyframes[i] for i in indices]
#         else:  # L == len(keyframes)
#             selected_keyframes = keyframes
        
#         cnt_cos_sim = 0
#         for keyframe in selected_keyframes:
#             half_T = T // 2

#             if T % 2 == 0:
#                 start = keyframe - half_T
#                 end = keyframe + half_T
#             else:
#                 start = keyframe - half_T
#                 end = keyframe + half_T + 1

#             # 범위 초과 시 조정 (T개 유지)
#             if start < 0:
#                 end += -start
#                 start = 0
#             elif end > num_frames:
#                 start -= (end - num_frames)
#                 end = num_frames

#             # 여전히 start가 음수일 수도 있음
#             start = max(start, 0)

#             # 마지막 체크: 길이가 정확히 T 아니면 강제로 맞춤
#             if end - start != T:
#                 end = start + T  # or start = end - T (선택지)
#                 if end > num_frames:
#                     end = num_frames
#                     start = end - T
#                     start = max(start, 0)
#             sampled_indices = np.arange(start, end)
            
#             # 샘플링한 인덱스로 클립 생성
#             clip = pose[sampled_indices]
#             clip_confidence = confidence[sampled_indices]
#             mean_confidence = np.mean(clip_confidence)
            
#             '''
#             Compute cos sim
#             '''
#             cos_sim = util.compute_pose_cosine_similarity(clip)
#             if np.any(cos_sim < 0.92):
#                 #print(f"Pose_len : {len(pose)},Keyframe_len : {len(keyframes)}")
#                 print(f'{video_id}:{sampled_indices}')
#                 cnt_cos_sim+=1
#                 if cnt_cos_sim == len(selected_keyframes):
#                     cnt+=1
#                 continue
#             if mean_confidence<args.confidence:
#                 print(video_id)
#                 continue
                
#             if len(clip) == T:
#                 '''
#                 여기서 스케일링해줘야됌.
#                 '''
#                 clip_normalized = normalized_keypoints(clip, scaler)
#                 all_clips.append(clip_normalized)
#                 class_metadata.append(f"{video_id}[{start},{end}]")
#             else:
#                 missing_video.append(video_id)

        
#         class_data.extend(all_clips)
#     # print(missing_video)
#     print(f"Number of missing video : {cnt}")
#     return class_data, class_metadata
def subsampling_considering_cos_sim(args, json_files):
    """
    Keyframe 중심 샘플링 + CSV 기반 필터링 (UCF101 전용 폴더+파일명 구조 반영)
    args.dataset == "UCF101" 인 경우,
      - args.json_path/<class_name>/*_result.json 구조에서 JSON 읽기
      - CSV 경로 args.anno_path/train.csv, val.csv 에 나온 video ID만 처리
    """
    scaler = MinMaxScaler((0,1))
    class_data = []
    class_metadata = []
    cnt_missing = 0
    L, T = args.num_subsequence, args.len_subsequence

    # 1) UCF101 전용: 클래스별 하위 폴더에서 JSON 수집
    if args.dataset == "UCF101":
        collected = []
        for class_dir in glob.glob(os.path.join(args.json_path, "*")):
            if not os.path.isdir(class_dir):
                continue
            collected += glob.glob(os.path.join(class_dir, "*_result.json"))
        print(f"[DEBUG] Found {len(collected)} JSON under {args.json_path}")
        json_files = collected

    # 2) CSV에서 valid_ids 수집
    valid_ids = set()
    for split in ("train","val"):
        csv_f = os.path.join(args.anno_path, f"{split}.csv")
        df = pd.read_csv(csv_f, header=None)
        for path in df[0].astype(str).tolist():
            name = os.path.basename(path)           # e.g. "v_WritingOnBoard_g07_c05.avi"
            video_id = os.path.splitext(name)[0]    # -> "v_WritingOnBoard_g07_c05"
            valid_ids.add(video_id)
    print(f"[DEBUG] CSV provided {len(valid_ids)} valid IDs")

    # 3) JSON 파일 필터링
    filtered = []
    for jf in json_files:
        base = os.path.basename(jf).replace("_result.json","")
        if args.dataset == "UCF101":
            cls = os.path.basename(os.path.dirname(jf))       # class_name e.g. "WritingOnBoard"
            # JSON basename might be "WritingOnBoard_v_...". strip prefix if present:
            if base.startswith(cls + "_"):
                vid = base[len(cls)+1:]
            else:
                vid = base
        else:
            vid = base
        if vid in valid_ids:
            filtered.append(jf)
    print(f"[DEBUG] After CSV filter: {len(filtered)}/{len(json_files)} JSON files")

    # 4) 만약 하나도 걸리지 않으면 전체 사용
    if not filtered:
        print("[WARN] CSV 필터 후 0개 → 전체 JSON 사용")
        filtered = json_files

    # 5) 실제 subsampling 수행
    for jf in tqdm(filtered, desc="Processing JSON files"):
        # 영상 ID 재추출 (for metadata)
        base = os.path.basename(jf).replace("_result.json","")
        if args.dataset == "UCF101":
            cls = os.path.basename(os.path.dirname(jf))
            vid = base[len(cls)+1:] if base.startswith(cls + "_") else base
        else:
            vid = base

        frames, missing_idx, orig_len = get_keypoints(jf)
        if not frames:
            cnt_missing += 1
            continue

        pose = np.stack(frames)[:,:,:2]       # (F,17,2)
        conf = np.stack(frames)[:,:,2]        # (F,17)
        F = pose.shape[0]
        if F < T:
            pose = util.repeat_to_min_length(pose, T)
            conf = util.repeat_to_min_length(conf, T)
            F = pose.shape[0]

        # keyframe 불러오기
        kf_path = os.path.join(args.keyframe_path, vid, "csvFile", f"{vid}.txt")
        if not os.path.exists(kf_path):
            cnt_missing += 1
            continue
        with open(kf_path) as f:
            keyframes = [int(x.strip()) for x in f]

        # missing index remap
        if missing_idx:
            valid_idx_map = [i for i in range(orig_len) if i not in missing_idx]
            keyframes = [valid_idx_map.index(k) for k in keyframes if k in valid_idx_map]
            if not keyframes:
                cnt_missing += 1
                continue

        # L 개 keyframe 선택
        if L > len(keyframes):
            sel = keyframes
        elif L < len(keyframes):
            idxs = np.linspace(0, len(keyframes)-1, L, dtype=int)
            sel = [keyframes[i] for i in idxs]
        else:
            sel = keyframes

        for k in sel:
            start = max(k - T//2, 0)
            end   = start + T
            if end > F:
                start = F - T
                end   = F
            clip = pose[start:end]
            clip_conf = conf[start:end]
            if clip.shape[0] != T:
                continue

            if clip_conf.mean() < args.confidence:
                continue
            if np.any(util.compute_pose_cosine_similarity(clip) < 0.92):
                continue

            # per-frame normalization
            clip_norm = np.stack([
                scaler.fit_transform(frame - frame.mean(axis=0))
                for frame in clip
            ], axis=0)
            class_data.append(clip_norm)
            class_metadata.append(f"{vid}[{start},{end}]")

    print(f"[INFO] Missing/skipped videos: {cnt_missing}")
    return class_data, class_metadata

def subsampling_wo_cos_sim(args, json_files):
    """Keyframe을 중심으로 일정 구간을 샘플링
    keyframe 개수에 따라 샘플링 개수 다르게
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    class_data = []
    class_metadata = []
    cnt = 0
    missing_video =[]
    L, T = args.num_subsequence, args.len_subsequence
    
    for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
        video_id = os.path.basename(json_file).replace("_result.json", "")
        frames_data, missing_index,og_num_frames = get_keypoints(json_file)
        
        
        
        
        all_clips = []

        if len(frames_data) == 0:
            print(f"Skipping {json_file} (No valid frames)")
            missing_video.append(json_file)
            cnt += 1
            continue
        pose = np.array(frames_data)[:, :,:2]
        confidence = np.array(frames_data)[:, :, 2] 
        num_frames = len(pose)
        if num_frames < T:
            pose = util.repeat_to_min_length(pose, T)
            confidence = util.repeat_to_min_length(confidence, T)
            print(f"Original frame len : {num_frames}")
            print(f"Expanding frames to {len(pose)}")
            num_frames = len(pose)
        
        keyframe_path = os.path.join(args.keyframe_path, video_id, "csvFile", f"{video_id}.txt")
        with open(keyframe_path, 'r') as f:
            keyframes = [int(line.strip()) for line in f.readlines()]
            
        # ✅ missing_index 있는 경우
        if len(missing_index) > 0:
            # 1. usable 인덱스 만들기
            valid_indices = [i for i in range(og_num_frames) if i not in missing_index]

            # 2. keyframes 중 usable 한 것만 남기고, frames_data 기준 인덱스로 재매핑
            remapped_keyframes = []
            for kf in keyframes:
                if kf in valid_indices:
                    remapped_keyframes.append(valid_indices.index(kf))  # usable index 기준으로 바꿔줌
                else:
                    print(f"Keyframe {kf} in {video_id} is missing and will be skipped.")

            keyframes = remapped_keyframes

            if len(keyframes) == 0:
                print(f"Skipping {video_id} (All keyframes are invalid after remapping)")
                cnt+=1
                continue
            
        # L, keyframe 개수 관계에 따른 처리
        selected_keyframes = []
        
        if L > len(keyframes):
            # L > keyframe: keyframe 개수만큼만
            selected_keyframes = keyframes
        elif L < len(keyframes):
            indices = np.linspace(0, len(keyframes)-1, L, dtype=int)
            selected_keyframes = [keyframes[i] for i in indices]
        else:  # L == len(keyframes)
            selected_keyframes = keyframes
        
        for keyframe in selected_keyframes:
            half_T = T // 2

            if T % 2 == 0:
                start = keyframe - half_T
                end = keyframe + half_T
            else:
                start = keyframe - half_T
                end = keyframe + half_T + 1

            # 범위 초과 시 조정 (T개 유지)
            if start < 0:
                end += -start
                start = 0
            elif end > num_frames:
                start -= (end - num_frames)
                end = num_frames

            # 여전히 start가 음수일 수도 있음
            start = max(start, 0)

            # 마지막 체크: 길이가 정확히 T 아니면 강제로 맞춤
            if end - start != T:
                end = start + T  # or start = end - T (선택지)
                if end > num_frames:
                    end = num_frames
                    start = end - T
                    start = max(start, 0)
            sampled_indices = np.arange(start, end)
            
            # 샘플링한 인덱스로 클립 생성
            clip = pose[sampled_indices]
            clip_confidence = confidence[sampled_indices]
            mean_confidence = np.mean(clip_confidence)
            
            '''
            Compute cos sim
            '''
                
            if len(clip) == T:
                '''
                여기서 스케일링해줘야됌.
                '''
                clip_normalized = normalized_keypoints(clip, scaler)
                all_clips.append(clip_normalized)
                class_metadata.append(f"{video_id}[{start},{end}]")
            else:
                missing_video.append(video_id)

        
        class_data.extend(all_clips)
    # print(missing_video)
    print(f"Number of missing video : {cnt}")
    return class_data, class_metadata

def subsampling_sliding_window(args, json_files):
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    class_data = []
    class_metadata = []
    missing_video = []
    cnt_invalid = 0
    
    L, T, stride = args.num_subsequence, args.len_subsequence, 1

    for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
        video_id = os.path.basename(json_file).replace("_result.json", "")
        frames_data, missing_index, og_num_frames = get_keypoints(json_file)

        if len(frames_data) == 0:
            print(f"Skipping {json_file} (No valid frames)")
            missing_video.append(json_file)
            cnt_invalid += 1
            continue

        pose = np.array(frames_data)[:, :, :2]       # (F, 17, 2)
        confidence = np.array(frames_data)[:, :, 2]  # (F, 17)
        num_frames = len(pose)

        if num_frames < T:
            pose = util.repeat_to_min_length(pose, T)
            confidence = util.repeat_to_min_length(confidence, T)
            print(f"Expanded {video_id}: {num_frames} → {len(pose)} frames")
            num_frames = len(pose)

        accepted_clips = []

        for i in range(0, num_frames - T + 1, stride):
            clip = pose[i:i+T]
            clip_conf = confidence[i:i+T]

            mean_conf = np.mean(clip_conf)
            cos_sim = util.compute_pose_cosine_similarity(clip)

            if np.any(cos_sim < 0.92):
                print(f"[cos_sim<0.92] {video_id}: [{i},{i+T}]")
                continue

            if mean_conf < args.confidence:
                print(f"[confidence<{args.confidence}] {video_id}")
                continue

            clip_normalized = normalized_keypoints(clip, scaler)
            accepted_clips.append(clip_normalized)
            class_metadata.append(f"{video_id}[{i},{i+T}]")

        if len(accepted_clips) == 0:
            cnt_invalid += 1
            missing_video.append(video_id)

        class_data.extend(accepted_clips)

    print(f"Number of invalid/missing videos: {cnt_invalid}")
    return class_data, class_metadata

def subsampling_non_overlapping(args, json_files):
    scaler = MinMaxScaler(feature_range=(0, 1))
    class_data = []
    class_metadata = []
    missing_video = []
    cnt = 0
    
    L, T = args.num_subsequence, args.len_subsequence
    threshold_conf = args.confidence  # ex. 0.9

    for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
        class_name = os.path.basename(os.path.dirname(json_file))
        video_id = os.path.basename(json_file).replace("_result.json", "")
        frames_data, missing_index,og_num_frames = get_keypoints(json_file)

        if len(frames_data) == 0:
            print(f"Skipping {json_file} (No valid frames)")
            missing_video.append(json_file)
            cnt += 1
            continue
        
        num_frames = len(frames_data)


        # 부족한 경우 확장
        if num_frames < L * T:
            expanded_frames = util.expand_array(frames_data, L * T)
            if expanded_frames is not None:
                frames_data = expanded_frames
                num_frames = len(frames_data)
            else:
                print(f"Skipping {json_file} (Unable to expand frames)")
                continue

        pose = np.array(frames_data)[:, :,:2]
        confidence = np.array(frames_data)[:, :, 2] 
        

        segment_length = num_frames // L
        all_clips = []

        for i in range(L):
            segment_start = i * segment_length
            segment_end = (i + 1) * segment_length

            sub_segment_length = max(segment_length // T, 1)
            sampled_indices = [
                segment_start + j * sub_segment_length + np.random.randint(0, sub_segment_length)
                for j in range(T)
            ]

            clip = pose[sampled_indices]
            clip_conf = confidence[sampled_indices]

            mean_conf = np.mean(clip_conf)
            cos_sim = util.compute_pose_cosine_similarity(clip)

            if np.any(cos_sim < 0.92):
                print(f"[cos_sim<0.92] {video_id} seg#{i}")
                continue
            if mean_conf < threshold_conf:
                print(f"[conf<{threshold_conf}] {video_id} seg#{i}")
                continue

            clip_normalized = normalized_keypoints(clip, scaler)
            all_clips.append(clip_normalized)
            class_metadata.append(f"{video_id}[{min(sampled_indices)},{max(sampled_indices)}]")

        class_data.extend(all_clips)

    return class_data, class_metadata

def Keypointset(args, output_path):
    processed_keypoints_path = os.path.join(output_path, "processed_keypoints.npy")
    if os.path.exists(processed_keypoints_path):
        print(f"✅ {processed_keypoints_path} 파일이 존재하므로 Keypointset()을 건너뜁니다.")
        return  
    class_list = util.load_class_list(args.anno_path)
    json_files = util.load_json_files(args.json_path, class_list, args.dataset)
    if args.subsampling_mode == "ver1":
        class_data, class_metadata = subsampling_ver1(args, json_files)
    elif args.subsampling_mode == "ver2":
        class_data, class_metadata = subsampling_ver2(args, json_files)
    elif args.subsampling_mode == "ver3":
        class_data, class_metadata = subsampling_ver3(args, json_files)
    elif args.subsampling_mode == "ver4":
        class_data, class_metadata = subsampling_ver4(args, json_files)
    elif args.subsampling_mode == "ver5":
        class_data, class_metadata = subsampling_ver5(args, json_files)
    elif args.subsampling_mode == "sim+conf":
        class_data, class_metadata = subsampling_considering_cos_sim(args, json_files)
    elif args.subsampling_mode == "wo_cos_sim":
        class_data, class_metadata = subsampling_wo_cos_sim(args, json_files)
    elif args.subsampling_mode == "sliding_window":
        class_data, class_metadata = subsampling_sliding_window(args, json_files)
    elif args.subsampling_mode == "non-overlapping":
        class_data, class_metadata = subsampling_non_overlapping(args, json_files)
    print(np.array(class_data).shape)
    print(np.array(class_metadata).shape)
    util.save_data(output_path, class_data, class_metadata)
