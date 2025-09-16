#!/usr/bin/env python3
"""
Schema Profiler - Web版
ブラウザベースの認知スキーマ測定ツール
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import numpy as np
import time
import json
import uuid
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
import plotly.graph_objs as go
import plotly.utils

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
CORS(app)

class TaskType(Enum):
    VISUAL_SEARCH = "visual_search"
    PATTERN_RECOGNITION = "pattern_recognition" 
    WORKING_MEMORY = "working_memory"
    STROOP_TEST = "stroop_test"

@dataclass
class TaskResult:
    task_type: TaskType
    accuracy: float
    reaction_time: float
    strategy_markers: Dict[str, Any]
    difficulty_level: int
    timestamp: float
    session_id: str

class CognitiveProfiler:
    """認知プロファイラーのメインクラス"""
    
    def __init__(self):
        self.results = {}  # session_id -> List[TaskResult]
    
    def add_result(self, session_id: str, result: TaskResult):
        """結果を記録"""
        if session_id not in self.results:
            self.results[session_id] = []
        self.results[session_id].append(result)
    
    def generate_visual_search_task(self, difficulty: int = 1) -> Dict[str, Any]:
        """視覚探索タスクを生成"""
        np.random.seed(int(time.time()))
        
        # 難易度に応じてアイテム数を調整
        target_count = 1
        distractor_count = 5 + (difficulty * 3)
        
        # ターゲットとディストラクターの色
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        target_color = np.random.choice(colors)
        distractor_colors = [c for c in colors if c != target_color]
        
        # アイテム生成
        items = []
        
        # ターゲット
        target_x = np.random.randint(50, 750)
        target_y = np.random.randint(50, 550)
        items.append({
            'id': 'target',
            'x': target_x,
            'y': target_y,
            'color': target_color,
            'shape': 'circle',
            'is_target': True
        })
        
        # ディストラクター
        for i in range(distractor_count):
            while True:
                x = np.random.randint(50, 750)
                y = np.random.randint(50, 550)
                
                # 他のアイテムと重複しないかチェック
                too_close = False
                for item in items:
                    if abs(x - item['x']) < 60 and abs(y - item['y']) < 60:
                        too_close = True
                        break
                
                if not too_close:
                    break
            
            items.append({
                'id': f'distractor_{i}',
                'x': x,
                'y': y,
                'color': np.random.choice(distractor_colors),
                'shape': 'circle',
                'is_target': False
            })
        
        return {
            'task_type': 'visual_search',
            'difficulty': difficulty,
            'target_color': target_color,
            'items': items,
            'instruction': f'{target_color}の円をクリックしてください'
        }
    
    def generate_pattern_task(self, difficulty: int = 1) -> Dict[str, Any]:
        """パターン認識タスクを生成"""
        np.random.seed(int(time.time()))
        
        pattern_length = 3 + difficulty
        pattern_types = ['arithmetic', 'geometric', 'fibonacci']
        pattern_type = np.random.choice(pattern_types)
        
        if pattern_type == 'arithmetic':
            start = np.random.randint(1, 10)
            diff = np.random.randint(1, 5)
            sequence = [start + i*diff for i in range(pattern_length)]
            next_val = start + pattern_length * diff
            
        elif pattern_type == 'geometric':
            start = np.random.randint(2, 5)
            ratio = np.random.randint(2, 3)
            sequence = [start * (ratio**i) for i in range(pattern_length)]
            next_val = start * (ratio**pattern_length)
            
        else:  # fibonacci
            sequence = [1, 1]
            for i in range(pattern_length-2):
                sequence.append(sequence[-1] + sequence[-2])
            next_val = sequence[-1] + sequence[-2]
        
        # 選択肢生成
        choices = [next_val]
        while len(choices) < 4:
            wrong_choice = next_val + np.random.randint(-10, 10)
            if wrong_choice not in choices and wrong_choice > 0:
                choices.append(wrong_choice)
        
        np.random.shuffle(choices)
        correct_answer = choices.index(next_val)
        
        return {
            'task_type': 'pattern_recognition',
            'difficulty': difficulty,
            'sequence': sequence,
            'choices': choices,
            'correct_answer': correct_answer,
            'pattern_type': pattern_type,
            'instruction': 'パターンを見つけて次の数を選んでください'
        }
    
    def generate_stroop_task(self, difficulty: int = 1) -> Dict[str, Any]:
        """ストループタスクを生成"""
        np.random.seed(int(time.time()))
        
        colors = ['赤', '青', '緑', '黄']
        color_mappings = {
            '赤': '#FF0000',
            '青': '#0000FF', 
            '緑': '#00AA00',
            '黄': '#FFAA00'
        }
        
        word = np.random.choice(colors)
        display_color = np.random.choice(colors)
        
        # 難易度に応じて一致・不一致の確率を調整
        if difficulty == 1:
            # 簡単：70%一致
            if np.random.random() < 0.7:
                display_color = word
        elif difficulty == 2:
            # 中：50%一致
            if np.random.random() < 0.5:
                display_color = word
        else:
            # 難しい：30%一致
            if np.random.random() < 0.3:
                display_color = word
        
        is_congruent = (word == display_color)
        
        return {
            'task_type': 'stroop_test',
            'difficulty': difficulty,
            'word': word,
            'color': color_mappings[display_color],
            'correct_answer': word,
            'is_congruent': is_congruent,
            'choices': colors,
            'instruction': '文字の意味（色名）を答えてください。表示色に惑わされないように。'
        }
    
    def generate_profile(self, session_id: str) -> Dict[str, Any]:
        """認知プロファイルを生成"""
        if session_id not in self.results or not self.results[session_id]:
            return {}
        
        results = self.results[session_id]
        
        # タスク別分析
        task_analysis = {}
        for task_type in TaskType:
            task_results = [r for r in results if r.task_type == task_type]
            if not task_results:
                continue
            
            accuracies = [r.accuracy for r in task_results]
            rts = [r.reaction_time for r in task_results if r.reaction_time < 999]
            
            task_analysis[task_type.value] = {
                'accuracy': np.mean(accuracies),
                'reaction_time': np.mean(rts) if rts else None,
                'consistency': 1.0 - np.std(accuracies) if len(accuracies) > 1 else 1.0,
                'improvement': self._calculate_improvement(task_results)
            }
        
        # 全体分析
        all_accuracies = [r.accuracy for r in results]
        all_rts = [r.reaction_time for r in results if r.reaction_time < 999]
        
        overall = {
            'total_trials': len(results),
            'overall_accuracy': np.mean(all_accuracies),
            'average_rt': np.mean(all_rts) if all_rts else None,
            'cognitive_flexibility': self._calculate_flexibility(results)
        }
        
        # 認知スタイル推定
        cognitive_style = self._analyze_cognitive_style(results)
        
        return {
            'session_id': session_id,
            'overall': overall,
            'task_analysis': task_analysis,
            'cognitive_style': cognitive_style,
            'timestamp': time.time()
        }
    
    def _calculate_improvement(self, task_results: List[TaskResult]) -> float:
        """学習効果を計算"""
        if len(task_results) < 4:
            return 0.0
        
        sorted_results = sorted(task_results, key=lambda x: x.timestamp)
        early_acc = np.mean([r.accuracy for r in sorted_results[:len(sorted_results)//2]])
        late_acc = np.mean([r.accuracy for r in sorted_results[len(sorted_results)//2:]])
        
        return late_acc - early_acc
    
    def _calculate_flexibility(self, results: List[TaskResult]) -> float:
        """認知的柔軟性を計算"""
        if len(results) < 4:
            return 0.5
        
        task_accuracies = {}
        for result in results:
            if result.task_type not in task_accuracies:
                task_accuracies[result.task_type] = []
            task_accuracies[result.task_type].append(result.accuracy)
        
        if len(task_accuracies) < 2:
            return 0.5
        
        task_means = [np.mean(accs) for accs in task_accuracies.values()]
        flexibility = 1.0 - np.std(task_means)
        return max(0.0, min(1.0, flexibility))
    
    def _analyze_cognitive_style(self, results: List[TaskResult]) -> Dict[str, str]:
        """認知スタイルを分析"""
        style = {
            'processing_speed': 'unknown',
            'accuracy_preference': 'unknown',
            'strategy_type': 'unknown'
        }
        
        if not results:
            return style
        
        all_rts = [r.reaction_time for r in results if r.reaction_time < 999]
        all_accs = [r.accuracy for r in results]
        
        if all_rts and all_accs:
            avg_rt = np.mean(all_rts)
            avg_acc = np.mean(all_accs)
            
            # 処理速度
            if avg_rt < 3:
                style['processing_speed'] = 'fast'
            elif avg_rt < 6:
                style['processing_speed'] = 'moderate'
            else:
                style['processing_speed'] = 'deliberate'
            
            # 正確性志向
            if avg_acc > 0.85:
                style['accuracy_preference'] = 'high_accuracy'
            elif avg_acc > 0.65:
                style['accuracy_preference'] = 'balanced'
            else:
                style['accuracy_preference'] = 'speed_focused'
            
            # 戦略タイプ
            if avg_rt > 5 and avg_acc > 0.8:
                style['strategy_type'] = 'analytical'
            elif avg_rt < 3 and avg_acc > 0.7:
                style['strategy_type'] = 'intuitive'
            else:
                style['strategy_type'] = 'adaptive'
        
        return style

# プロファイラーインスタンス
profiler = CognitiveProfiler()

@app.route('/')
def index():
    """メインページ"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('index.html')

@app.route('/health')
def health():
    """ヘルスチェック"""
    return jsonify({'status': 'ok', 'timestamp': time.time()})

@app.route('/api/task/<task_type>/<int:difficulty>')
def get_task(task_type: str, difficulty: int):
    """タスクを生成"""
    try:
        if task_type == 'visual_search':
            task_data = profiler.generate_visual_search_task(difficulty)
        elif task_type == 'pattern_recognition':
            task_data = profiler.generate_pattern_task(difficulty)
        elif task_type == 'stroop_test':
            task_data = profiler.generate_stroop_task(difficulty)
        else:
            return jsonify({'error': 'Unknown task type'}), 400
        
        return jsonify({
            'success': True,
            'task': task_data,
            'start_time': time.time()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/submit', methods=['POST'])
def submit_result():
    """結果を提出"""
    try:
        data = request.get_json()
        session_id = session.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'No session ID'}), 400
        
        # TaskResult作成
        result = TaskResult(
            task_type=TaskType(data['task_type']),
            accuracy=float(data['accuracy']),
            reaction_time=float(data['reaction_time']),
            strategy_markers=data.get('strategy_markers', {}),
            difficulty_level=int(data['difficulty']),
            timestamp=time.time(),
            session_id=session_id
        )
        
        profiler.add_result(session_id, result)
        
        return jsonify({
            'success': True,
            'total_results': len(profiler.results.get(session_id, []))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile')
def get_profile():
    """プロファイルを取得"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session ID'}), 400
        
        profile = profiler.generate_profile(session_id)
        
        return jsonify({
            'success': True,
            'profile': profile
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/profile')
def profile_view():
    """プロファイル表示ページ"""
    return render_template('profile.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)