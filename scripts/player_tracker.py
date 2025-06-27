import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import json
from collections import defaultdict, deque

class PlayerTracker:
    def __init__(self, max_disappeared=30, max_distance=100, similarity_threshold=0.3):
        self.next_id = 0
        self.objects = {}  # Active tracks
        self.disappeared = {}  # Disappeared tracks with frame count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.similarity_threshold = similarity_threshold
        
        # Feature storage for re-identification
        self.feature_history = defaultdict(list)
        self.appearance_features = {}
        self.track_history = defaultdict(list)
        
    def extract_appearance_features(self, frame, bbox):
        """Extract appearance features from player region"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(48)  # Return zero vector for invalid bbox
        
        # Extract player region
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return np.zeros(48)
        
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(player_region, cv2.COLOR_BGR2HSV)
        
        # Calculate color histogram
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # Normalize histograms
        hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
        hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
        hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)
        
        # Combine histograms
        features = np.concatenate([hist_h, hist_s, hist_v])
        return features
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors"""
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        # Reshape for cosine similarity
        f1 = np.array(features1).reshape(1, -1)
        f2 = np.array(features2).reshape(1, -1)
        
        return cosine_similarity(f1, f2)[0][0]
    
    def register(self, centroid, bbox, features, frame_num):
        """Register a new object"""
        self.objects[self.next_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'features': features,
            'last_seen': frame_num,
            'track_length': 1
        }
        self.feature_history[self.next_id].append(features)
        self.track_history[self.next_id].append((frame_num, centroid, bbox))
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections, frame, frame_num):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # Extract centroids and features from detections
        input_centroids = []
        input_bboxes = []
        input_features = []
        
        for detection in detections:
            bbox = detection['bbox']
            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            features = self.extract_appearance_features(frame, bbox)
            
            input_centroids.append(centroid)
            input_bboxes.append(bbox)
            input_features.append(features)
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], input_features[i], frame_num)
        else:
            # Compute distance and similarity matrices
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id]['centroid'] for obj_id in object_ids]
            
            # Distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - 
                             np.array(input_centroids), axis=2)
            
            # Similarity matrix for appearance
            S = np.zeros((len(object_ids), len(input_features)))
            for i, obj_id in enumerate(object_ids):
                obj_features = self.objects[obj_id]['features']
                for j, input_feat in enumerate(input_features):
                    S[i, j] = self.calculate_similarity(obj_features, input_feat)
            
            # Combined score (lower distance + higher similarity = better match)
            combined_score = D - (S * 100)  # Scale similarity to balance with distance
            
            # Find optimal assignment using Hungarian algorithm approximation
            rows = combined_score.min(axis=1).argsort()
            cols = combined_score.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if (combined_score[row, col] <= self.max_distance and 
                    S[row, col] >= self.similarity_threshold):
                    
                    object_id = object_ids[row]
                    self.objects[object_id]['centroid'] = input_centroids[col]
                    self.objects[object_id]['bbox'] = input_bboxes[col]
                    self.objects[object_id]['features'] = input_features[col]
                    self.objects[object_id]['last_seen'] = frame_num
                    self.objects[object_id]['track_length'] += 1
                    
                    # Update feature history
                    self.feature_history[object_id].append(input_features[col])
                    if len(self.feature_history[object_id]) > 10:  # Keep last 10 features
                        self.feature_history[object_id].pop(0)
                    
                    # Update track history
                    self.track_history[object_id].append((frame_num, input_centroids[col], input_bboxes[col]))
                    
                    self.disappeared[object_id] = 0
                    used_row_indices.add(row)
                    used_col_indices.add(col)
            
            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, combined_score.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, combined_score.shape[1])).difference(used_col_indices)
            
            # Mark unmatched objects as disappeared
            for row in unused_row_indices:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects or try re-identification
            for col in unused_col_indices:
                # Try re-identification with disappeared objects
                reidentified = False
                best_match_id = None
                best_similarity = 0
                
                # Check against recently disappeared objects
                for obj_id in list(self.feature_history.keys()):
                    if obj_id not in self.objects:  # Disappeared object
                        # Calculate average similarity with historical features
                        similarities = []
                        for hist_features in self.feature_history[obj_id][-5:]:  # Last 5 features
                            sim = self.calculate_similarity(hist_features, input_features[col])
                            similarities.append(sim)
                        
                        avg_similarity = np.mean(similarities) if similarities else 0
                        
                        if avg_similarity > best_similarity and avg_similarity > 0.4:  # Higher threshold for re-ID
                            best_similarity = avg_similarity
                            best_match_id = obj_id
                
                if best_match_id is not None:
                    # Re-identify the object
                    self.objects[best_match_id] = {
                        'centroid': input_centroids[col],
                        'bbox': input_bboxes[col],
                        'features': input_features[col],
                        'last_seen': frame_num,
                        'track_length': self.objects.get(best_match_id, {}).get('track_length', 0) + 1
                    }
                    self.disappeared[best_match_id] = 0
                    
                    # Update histories
                    self.feature_history[best_match_id].append(input_features[col])
                    self.track_history[best_match_id].append((frame_num, input_centroids[col], input_bboxes[col]))
                    
                    reidentified = True
                
                if not reidentified:
                    # Register as new object
                    self.register(input_centroids[col], input_bboxes[col], input_features[col], frame_num)
        
        return self.objects
    
    def get_track_statistics(self):
        """Get tracking statistics"""
        stats = {
            'total_tracks': len(self.track_history),
            'active_tracks': len(self.objects),
            'track_lengths': {},
            'reidentification_events': 0
        }
        
        for track_id, history in self.track_history.items():
            stats['track_lengths'][track_id] = len(history)
            
            # Count gaps in tracking (potential re-identifications)
            frame_nums = [h[0] for h in history]
            gaps = 0
            for i in range(1, len(frame_nums)):
                if frame_nums[i] - frame_nums[i-1] > 5:  # Gap of more than 5 frames
                    gaps += 1
            stats['reidentification_events'] += gaps
        
        return stats
