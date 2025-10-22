import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

class HierarchicalClustering_:
    """계층적 클러스터링 클래스 - Single Linkage"""
    
    def __init__(self):
        self.clusters = []
        self.merge_history = []
        self.all_clusters = {}
        
    def euclidean_distance(self, point1, point2):
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
    
    def calculate_cluster_distance_single(self, cluster1, cluster2):
        min_distance = float('inf')
        for point1 in cluster1['points']:
            for point2 in cluster2['points']:
                distance = self.euclidean_distance(point1, point2)
                if distance < min_distance:
                    min_distance = distance
        return min_distance
    
    def fit(self, data):
        print("\n=== 계층적 클러스터링 시작 (Single Linkage) ===\n")
        self.clusters = []
        for i, point in enumerate(data):
            cluster = {
                'id': i,
                'points': [point],
                'level': 0,
                'children': [],
                'height': 0, # 초기 높이는 0
                'original_indices': [i]
            }
            self.clusters.append(cluster)
            self.all_clusters[i] = cluster
            
        print(f"초기 클러스터 개수: {len(self.clusters)}개")
        
        cluster_id_counter = len(data)
        step = 1
        
        while len(self.clusters) > 1:
            min_distance = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    distance = self.calculate_cluster_distance_single(
                        self.clusters[i], self.clusters[j]
                    )
                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j
            
            cluster1 = self.clusters[merge_i]
            cluster2 = self.clusters[merge_j]
            
            new_cluster = {
                'id': cluster_id_counter,
                'points': cluster1['points'] + cluster2['points'],
                'level': max(cluster1['level'], cluster2['level']) + 1,
                'children': [cluster1['id'], cluster2['id']],
                'height': min_distance,
                'original_indices': cluster1['original_indices'] + cluster2['original_indices']
            }
            
            self.merge_history.append({
                'step': step,
                'cluster1_id': cluster1['id'],
                'cluster2_id': cluster2['id'],
                'new_cluster_id': cluster_id_counter,
                'distance': min_distance,
                'size': len(new_cluster['points'])
            })
            
            self.all_clusters[cluster_id_counter] = new_cluster
            
            if merge_i > merge_j:
                del self.clusters[merge_i]
                del self.clusters[merge_j]
            else:
                del self.clusters[merge_j]
                del self.clusters[merge_i]
            
            self.clusters.append(new_cluster)
            cluster_id_counter += 1
            step += 1
        
        print(f"\n=== 클러스터링 완료! 총 {step-1}번의 병합 수행 ===\n")

    def plot_dendrogram(self, data, num_major_clusters=3, save_path=None):
        """
        덴드로그램 시각화
        """
        if not self.merge_history:
            print("먼저 fit() 메서드를 실행하세요.")
            return

        fig, ax = plt.subplots(figsize=(14, 10))
        cluster_positions = {}
        next_x_position = 0

        # 1. 리프 노드 위치 설정
        for i in range(len(data)):
            cluster_positions[i] = next_x_position
            next_x_position += 10
            ax.text(cluster_positions[i], -0.05, f'P{i+1}', ha='center', va='top', fontsize=9, fontweight='bold')
            point = data[i]
            ax.text(cluster_positions[i], -0.15, f'({point[0]:.1f},{point[1]:.1f})', ha='center', va='top', fontsize=7, color='gray')

        # 2. 주요 클러스터 색상 할당
        cluster_colors = self._assign_cluster_colors(num_major_clusters)
        colors_for_legend = self._generate_colors(num_major_clusters)

        # 3. 덴드로그램 그리기
        for merge in self.merge_history:
            c1_id, c2_id = merge['cluster1_id'], merge['cluster2_id']
            new_id = merge['new_cluster_id']
            height = merge['distance']

            x1, x2 = cluster_positions[c1_id], cluster_positions[c2_id]
            x_new = (x1 + x2) / 2
            cluster_positions[new_id] = x_new

            c1_height = self.all_clusters.get(c1_id, {}).get('height', 0)
            c2_height = self.all_clusters.get(c2_id, {}).get('height', 0)

            # 색상 결정: 병합으로 생성된 새 클러스터의 ID를 기준으로 색상을 가져옴
            color = cluster_colors.get(new_id, 'gray')
            linewidth = 2.5

            # 수직선 및 수평선 그리기
            ax.plot([x1, x1], [c1_height, height], color=color, linewidth=linewidth)
            ax.plot([x2, x2], [c2_height, height], color=color, linewidth=linewidth)
            ax.plot([x1, x2], [height, height], color=color, linewidth=linewidth)

            ax.text(x_new, height, f'{height:.2f}', ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # 그래프 설정
        ax.set_xlabel('데이터 포인트', fontsize=12, fontweight='bold')
        ax.set_ylabel('거리 (Distance)', fontsize=12, fontweight='bold')
        ax.set_title('개선된 계층적 클러스터링 덴드로그램', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=-0.3)
        ax.set_xlim(-5, next_x_position - 5)

        # 범례 추가
        legend_elements = []
        for i in range(num_major_clusters):
            legend_elements.append(mpatches.Patch(
                color=colors_for_legend[i], 
                label=f'주요 클러스터 {i+1}'
            ))
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"덴드로그램 저장됨: {save_path}")
            plt.close()
        else:
            plt.show()


    def _get_clusters_at_step(self, step, num_points):
        clusters = {i: [i] for i in range(num_points)}
        for i in range(min(step, len(self.merge_history))):
            merge = self.merge_history[i]
            cluster1_points = clusters.get(merge['cluster1_id'], [])
            cluster2_points = clusters.get(merge['cluster2_id'], [])
            new_cluster_points = cluster1_points + cluster2_points
            if merge['cluster1_id'] in clusters:
                del clusters[merge['cluster1_id']]
            if merge['cluster2_id'] in clusters:
                del clusters[merge['cluster2_id']]
            clusters[merge['new_cluster_id']] = new_cluster_points
        return list(clusters.values())

    def _generate_colors(self, n):
        """n개의 구별되는 색상 생성"""
        # Use a standard colormap for good color distinction
        colors = plt.cm.get_cmap('tab10', n)
        return [colors(i) for i in range(n)]
    
    def _assign_cluster_colors(self, num_major_clusters):
        """클러스터별 색상 할당"""
        if num_major_clusters <= 0 or not self.merge_history:
            return {}
            
        colors = self._generate_colors(num_major_clusters)
        cluster_colors = {}
        
        # 마지막 (num_major_clusters - 1)개의 병합이 주요 클러스터를 결정
        if len(self.merge_history) < num_major_clusters -1:
            return {}

        # 주요 클러스터들의 원본 ID를 찾음
        major_cluster_children = []
        if num_major_clusters == 1:
            if self.merge_history:
                major_cluster_children.append(self.merge_history[-1]['new_cluster_id'])
        else:
            top_merges = self.merge_history[-(num_major_clusters - 1):]
            top_level_ids = {m['new_cluster_id'] for m in top_merges}
            all_children = set()
            for m in top_merges:
                all_children.add(m['cluster1_id'])
                all_children.add(m['cluster2_id'])
            major_cluster_children = list(all_children - top_level_ids)

        # 각 주요 클러스터와 그 자손들에게 색상을 전파
        for i, cluster_id in enumerate(major_cluster_children):
            color = colors[i % len(colors)]
            def propagate_color(c_id):
                cluster_colors[c_id] = color
                cluster = self.all_clusters.get(c_id, {})
                for child_id in cluster.get('children', []):
                    propagate_color(child_id)
            propagate_color(cluster_id)
            
        return cluster_colors

def generate_sample_data():
    data = []
    cluster1_center = (2, 2)
    for _ in range(4):
        x = cluster1_center[0] + random.uniform(-1, 1)
        y = cluster1_center[1] + random.uniform(-1, 1)
        data.append([x, y])
    cluster2_center = (8, 8)
    for _ in range(4):
        x = cluster2_center[0] + random.uniform(-1, 1)
        y = cluster2_center[1] + random.uniform(-1, 1)
        data.append([x, y])
    cluster3_center = (5, 5)
    for _ in range(3):
        x = cluster3_center[0] + random.uniform(-0.8, 0.8)
        y = cluster3_center[1] + random.uniform(-0.8, 0.8)
        data.append([x, y])
    return data

def main():
    print("="*60)
    print("계층적 클러스터링 - Single Linkage")
    print("="*60)
    data = generate_sample_data()
    print(f"\n📊 생성된 데이터 포인트: {len(data)}개")
    print("\n데이터 좌표:")
    for i, point in enumerate(data):
        print(f"  P{i+1}: ({point[0]:.2f}, {point[1]:.2f})")
    hc = HierarchicalClustering_()
    hc.fit(data)
    print("\n📝 병합 히스토리:")
    print("-" * 50)
    for merge in hc.merge_history:
        print(f"Step {merge['step']}: "
              f"클러스터 {merge['cluster1_id']} + {merge['cluster2_id']} "
              f"→ 클러스터 {merge['new_cluster_id']} "
              f"(거리: {merge['distance']:.3f}, 크기: {merge['size']}개)")
        
    hc.plot_dendrogram(data, save_path='dendrogram.png')

    print("dendrogram.png: 향상된 덴드로그램")


if __name__ == "__main__":
    main()
