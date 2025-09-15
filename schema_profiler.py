#!/usr/bin/env python3
"""
認知スキーマ・プロファイラー
レベルの認知特性を多角的に測定・可視化するツール
"""

import pygame
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import random
from enum import Enum

class TaskType(Enum):
    VISUAL_SEARCH = "visual_search"
    PATTERN_RECOGNITION = "pattern_recognition" 
    WORKING_MEMORY = "working_memory"
    ATTENTION_SWITCHING = "attention_switching"
    SPATIAL_REASONING = "spatial_reasoning"

@dataclass
class TaskResult:
    task_type: TaskType
    accuracy: float
    reaction_time: float
    strategy_markers: Dict[str, Any]
    difficulty_level: int
    timestamp: float

class CognitiveProfiler:
    """認知スキーマの特性を測定する統合プラットフォーム"""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Schema Profiler - 認知特性測定")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)
        
        # 色定義
        self.colors = {
            'bg': (40, 40, 50),
            'primary': (100, 150, 250),
            'secondary': (150, 100, 250),
            'success': (100, 250, 150),
            'warning': (250, 200, 100),
            'error': (250, 100, 100),
            'text': (220, 220, 220),
            'white': (255, 255, 255)
        }
        
        self.results: List[TaskResult] = []
        self.current_task = None
        
    def visual_search_task(self, difficulty=1) -> TaskResult:
        """視覚探索タスク - 注意の選択性と効率性を測定"""
        
        # 難易度に応じてターゲット・ディストラクター数を調整
        target_count = 1
        distractor_count = 5 + (difficulty * 3)
        
        # ターゲット設定
        target_color = (255, 100, 100)  # 赤
        distractor_colors = [(100, 255, 100), (100, 100, 255), (255, 255, 100)]
        
        objects = []
        
        # ターゲット配置
        tx, ty = random.randint(100, self.width-100), random.randint(100, self.height-100)
        objects.append(('target', tx, ty, target_color))
        
        # ディストラクター配置
        for _ in range(distractor_count):
            while True:
                dx, dy = random.randint(100, self.width-100), random.randint(100, self.height-100)
                if abs(dx-tx) > 60 or abs(dy-ty) > 60:  # ターゲットと重複回避
                    break
            color = random.choice(distractor_colors)
            objects.append(('distractor', dx, dy, color))
        
        # タスク実行
        start_time = time.time()
        response_time = None
        correct = False
        click_sequence = []  # 戦略分析用
        
        self.screen.fill(self.colors['bg'])
        instruction = "赤い円をクリックしてください"
        text_surface = self.font.render(instruction, True, self.colors['text'])
        self.screen.blit(text_surface, (10, 10))
        
        # オブジェクト描画
        for obj_type, x, y, color in objects:
            pygame.draw.circle(self.screen, color, (x, y), 20)
            if obj_type == 'target':
                pygame.draw.circle(self.screen, self.colors['white'], (x, y), 22, 2)
        
        pygame.display.flip()
        
        running = True
        while running and response_time is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    click_sequence.append((mx, my, time.time() - start_time))
                    
                    # ターゲット判定
                    for obj_type, x, y, color in objects:
                        if abs(mx-x) < 25 and abs(my-y) < 25:
                            response_time = time.time() - start_time
                            correct = (obj_type == 'target')
                            running = False
                            break
        
        # 戦略分析
        strategy_markers = {
            'click_count': len(click_sequence),
            'search_pattern': self._analyze_search_pattern(click_sequence),
            'hesitation_time': click_sequence[0][2] if click_sequence else 0
        }
        
        accuracy = 1.0 if correct else 0.0
        return TaskResult(
            TaskType.VISUAL_SEARCH, accuracy, response_time or 999,
            strategy_markers, difficulty, time.time()
        )
    
    def pattern_recognition_task(self, difficulty=1) -> TaskResult:
        """パターン認識タスク - 抽象的思考と規則発見能力を測定"""
        
        # 難易度に応じてパターンの複雑さを調整
        pattern_length = 3 + difficulty
        
        # 数列パターン生成（等差数列、等比数列、フィボナッチ風など）
        pattern_types = ['arithmetic', 'geometric', 'fibonacci', 'prime']
        pattern_type = random.choice(pattern_types)
        
        if pattern_type == 'arithmetic':
            start = random.randint(1, 10)
            diff = random.randint(1, 5)
            sequence = [start + i*diff for i in range(pattern_length)]
            next_val = start + pattern_length * diff
        elif pattern_type == 'geometric':
            start = random.randint(2, 5)
            ratio = random.randint(2, 3)
            sequence = [start * (ratio**i) for i in range(pattern_length)]
            next_val = start * (ratio**pattern_length)
        else:  # 簡単なフィボナッチ風
            sequence = [1, 1]
            for i in range(pattern_length-2):
                sequence.append(sequence[-1] + sequence[-2])
            next_val = sequence[-1] + sequence[-2]
        
        # 選択肢生成
        choices = [next_val]
        while len(choices) < 4:
            wrong_choice = next_val + random.randint(-10, 10)
            if wrong_choice not in choices and wrong_choice > 0:
                choices.append(wrong_choice)
        
        random.shuffle(choices)
        correct_idx = choices.index(next_val)
        
        # 表示
        start_time = time.time()
        response_time = None
        correct = False
        
        self.screen.fill(self.colors['bg'])
        
        # パターン表示
        pattern_text = "パターンを見つけて次の数を選んでください:"
        text_surface = self.font.render(pattern_text, True, self.colors['text'])
        self.screen.blit(text_surface, (50, 100))
        
        sequence_text = " → ".join(map(str, sequence)) + " → ?"
        seq_surface = self.large_font.render(sequence_text, True, self.colors['primary'])
        self.screen.blit(seq_surface, (50, 150))
        
        # 選択肢表示
        for i, choice in enumerate(choices):
            y_pos = 250 + i * 60
            choice_rect = pygame.Rect(50, y_pos, 100, 40)
            pygame.draw.rect(self.screen, self.colors['secondary'], choice_rect)
            choice_text = str(choice)
            text_surface = self.font.render(choice_text, True, self.colors['white'])
            text_rect = text_surface.get_rect(center=choice_rect.center)
            self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()
        
        running = True
        while running and response_time is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    
                    for i, choice in enumerate(choices):
                        y_pos = 250 + i * 60
                        choice_rect = pygame.Rect(50, y_pos, 100, 40)
                        if choice_rect.collidepoint(mx, my):
                            response_time = time.time() - start_time
                            correct = (i == correct_idx)
                            running = False
                            break
        
        strategy_markers = {
            'pattern_type': pattern_type,
            'pattern_length': pattern_length,
            'response_confidence': 1.0 - min(response_time or 999, 10) / 10
        }
        
        accuracy = 1.0 if correct else 0.0
        return TaskResult(
            TaskType.PATTERN_RECOGNITION, accuracy, response_time or 999,
            strategy_markers, difficulty, time.time()
        )
    
    def working_memory_task(self, difficulty=1) -> TaskResult:
        """ワーキングメモリタスク - 短期記憶と情報処理の同時実行能力を測定"""
        
        # N-back task風
        n_back = min(1 + difficulty, 4)  # 1-back to 4-back
        sequence_length = 10 + difficulty * 2
        
        # 刺激系列生成
        positions = [(200, 200), (400, 200), (600, 200), (200, 400), (400, 400), (600, 400)]
        sequence = [random.choice(positions) for _ in range(sequence_length)]
        
        # 正解の位置を記録
        target_positions = []
        for i in range(n_back, len(sequence)):
            if sequence[i] == sequence[i - n_back]:
                target_positions.append(i)
        
        start_time = time.time()
        responses = []
        current_idx = 0
        
        running = True
        stimulus_timer = 0
        stimulus_duration = 1000  # 1秒
        isi_duration = 500  # 0.5秒の間隔
        
        while running and current_idx < len(sequence):
            dt = self.clock.tick(60)
            stimulus_timer += dt
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        responses.append((current_idx, time.time() - start_time))
            
            self.screen.fill(self.colors['bg'])
            
            # 指示表示
            instruction = f"{n_back}-back: 刺激が{n_back}個前と同じ位置ならスペースキー"
            text_surface = self.font.render(instruction, True, self.colors['text'])
            self.screen.blit(text_surface, (50, 50))
            
            progress_text = f"進行: {current_idx + 1}/{len(sequence)}"
            progress_surface = self.font.render(progress_text, True, self.colors['text'])
            self.screen.blit(progress_surface, (50, 80))
            
            # 刺激表示
            if stimulus_timer < stimulus_duration:
                pos = sequence[current_idx]
                pygame.draw.circle(self.screen, self.colors['primary'], pos, 30)
            elif stimulus_timer >= stimulus_duration + isi_duration:
                current_idx += 1
                stimulus_timer = 0
            
            pygame.display.flip()
        
        # 正解率計算
        correct_responses = 0
        false_alarms = 0
        
        response_indices = [r[0] for r in responses]
        
        for target_pos in target_positions:
            if target_pos in response_indices:
                correct_responses += 1
        
        false_alarms = len([r for r in response_indices if r not in target_positions])
        
        hits = correct_responses
        misses = len(target_positions) - hits
        
        accuracy = hits / max(len(target_positions), 1) if target_positions else 1.0
        
        avg_response_time = np.mean([r[1] for r in responses]) if responses else 999
        
        strategy_markers = {
            'n_back_level': n_back,
            'hits': hits,
            'misses': misses,
            'false_alarms': false_alarms,
            'response_consistency': 1.0 - (false_alarms / max(len(responses), 1))
        }
        
        return TaskResult(
            TaskType.WORKING_MEMORY, accuracy, avg_response_time,
            strategy_markers, difficulty, time.time()
        )
    
    def _analyze_search_pattern(self, clicks: List[Tuple[int, int, float]]) -> str:
        """視覚探索の戦略パターンを分析"""
        if len(clicks) < 2:
            return "single_click"
        
        # 探索パターンの分析
        distances = []
        for i in range(1, len(clicks)):
            dx = clicks[i][0] - clicks[i-1][0]
            dy = clicks[i][1] - clicks[i-1][1]
            distances.append(np.sqrt(dx*dx + dy*dy))
        
        avg_distance = np.mean(distances)
        
        if avg_distance < 100:
            return "local_search"  # 局所探索
        elif avg_distance > 300:
            return "global_search"  # 大域探索
        else:
            return "systematic_search"  # 系統的探索
    
    def run_battery(self, num_trials_per_task=3):
        """認知バッテリー全体を実行"""
        print("認知スキーマ・プロファイラー開始")
        print("複数のタスクであなたの認知特性を測定します")
        
        tasks = [
            (self.visual_search_task, "視覚探索"),
            (self.pattern_recognition_task, "パターン認識"),
            (self.working_memory_task, "ワーキングメモリ")
        ]
        
        for task_func, task_name in tasks:
            print(f"\n{task_name}タスクを開始します...")
            
            for trial in range(num_trials_per_task):
                for difficulty in range(1, 4):  # 難易度1-3
                    result = task_func(difficulty)
                    if result:
                        self.results.append(result)
                        print(f"  試行 {trial+1}, 難易度 {difficulty}: 正答率 {result.accuracy:.2f}, RT {result.reaction_time:.3f}s")
        
        pygame.quit()
        return self.generate_profile()
    
    def generate_profile(self) -> Dict[str, Any]:
        """認知プロファイルを生成"""
        if not self.results:
            return {}
        
        profile = {
            'overall_metrics': {},
            'task_specific': {},
            'cognitive_style': {},
            'recommendations': []
        }
        
        # タスク別分析
        for task_type in TaskType:
            task_results = [r for r in self.results if r.task_type == task_type]
            if not task_results:
                continue
            
            accuracies = [r.accuracy for r in task_results]
            rts = [r.reaction_time for r in task_results if r.reaction_time < 999]
            
            profile['task_specific'][task_type.value] = {
                'accuracy': np.mean(accuracies),
                'reaction_time': np.mean(rts) if rts else None,
                'consistency': 1.0 - np.std(accuracies),
                'learning_effect': self._calculate_learning_effect(task_results)
            }
        
        # 認知スタイル分析
        profile['cognitive_style'] = self._analyze_cognitive_style()
        
        # 全体的メトリクス
        all_accuracies = [r.accuracy for r in self.results]
        all_rts = [r.reaction_time for r in self.results if r.reaction_time < 999]
        
        profile['overall_metrics'] = {
            'general_accuracy': np.mean(all_accuracies),
            'processing_speed': np.mean(all_rts) if all_rts else None,
            'cognitive_flexibility': self._calculate_flexibility(),
            'profile_timestamp': time.time()
        }
        
        return profile
    
    def _calculate_learning_effect(self, task_results: List[TaskResult]) -> float:
        """学習効果を計算"""
        if len(task_results) < 2:
            return 0.0
        
        # 時系列での精度向上を測定
        sorted_results = sorted(task_results, key=lambda x: x.timestamp)
        early_acc = np.mean([r.accuracy for r in sorted_results[:len(sorted_results)//2]])
        late_acc = np.mean([r.accuracy for r in sorted_results[len(sorted_results)//2:]])
        
        return late_acc - early_acc
    
    def _calculate_flexibility(self) -> float:
        """認知的柔軟性を計算"""
        if len(self.results) < 2:
            return 0.0
        
        # タスク間のパフォーマンス変動から柔軟性を推定
        task_accuracies = {}
        for result in self.results:
            if result.task_type not in task_accuracies:
                task_accuracies[result.task_type] = []
            task_accuracies[result.task_type].append(result.accuracy)
        
        # タスク間の相関から柔軟性を推定
        flexibility = 1.0 - np.std([np.mean(accs) for accs in task_accuracies.values()])
        return max(0.0, min(1.0, flexibility))
    
    def _analyze_cognitive_style(self) -> Dict[str, Any]:
        """認知スタイルを分析"""
        style = {
            'processing_style': 'unknown',
            'attention_style': 'unknown',
            'strategy_preference': 'unknown'
        }
        
        # 視覚探索結果から注意スタイルを推定
        vs_results = [r for r in self.results if r.task_type == TaskType.VISUAL_SEARCH]
        if vs_results:
            search_patterns = [r.strategy_markers.get('search_pattern', '') for r in vs_results]
            
            if search_patterns.count('systematic_search') > len(search_patterns) // 2:
                style['attention_style'] = 'systematic'
            elif search_patterns.count('local_search') > len(search_patterns) // 2:
                style['attention_style'] = 'focused'
            else:
                style['attention_style'] = 'flexible'
        
        # パターン認識から処理スタイルを推定
        pr_results = [r for r in self.results if r.task_type == TaskType.PATTERN_RECOGNITION]
        if pr_results:
            avg_rt = np.mean([r.reaction_time for r in pr_results if r.reaction_time < 999])
            avg_acc = np.mean([r.accuracy for r in pr_results])
            
            if avg_rt < 5 and avg_acc > 0.8:
                style['processing_style'] = 'intuitive'
            elif avg_rt > 10 and avg_acc > 0.8:
                style['processing_style'] = 'analytical'
            else:
                style['processing_style'] = 'balanced'
        
        return style

def main():
    """メイン実行関数"""
    profiler = CognitiveProfiler()
    
    print("=== 認知スキーマ・プロファイラー ===")
    print("あなたの認知特性を多角的に測定します")
    
    # バッテリー実行
    profile = profiler.run_battery(num_trials_per_task=2)
    
    # 結果表示
    print("\n" + "="*50)
    print("あなたの認知プロファイル")
    print("="*50)
    
    if profile:
        print(f"\n【総合スコア】")
        overall = profile.get('overall_metrics', {})
        print(f"  全体的正答率: {overall.get('general_accuracy', 0):.2%}")
        print(f"  処理速度: {overall.get('processing_speed', 0):.3f}秒")
        print(f"  認知的柔軟性: {overall.get('cognitive_flexibility', 0):.2f}")
        
        print(f"\n【認知スタイル】")
        style = profile.get('cognitive_style', {})
        print(f"  処理スタイル: {style.get('processing_style', 'unknown')}")
        print(f"  注意スタイル: {style.get('attention_style', 'unknown')}")
        
        print(f"\n【タスク別詳細】")
        for task_name, metrics in profile.get('task_specific', {}).items():
            print(f"  {task_name}:")
            print(f"    正答率: {metrics.get('accuracy', 0):.2%}")
            print(f"    反応時間: {metrics.get('reaction_time', 0):.3f}秒")
            print(f"    一貫性: {metrics.get('consistency', 0):.2f}")
            print(f"    学習効果: {metrics.get('learning_effect', 0):.2f}")
        
        # プロファイルをJSONで保存
        with open('cognitive_profile.json', 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n詳細なプロファイルを 'cognitive_profile.json' に保存しました")

if __name__ == "__main__":
    main()