import math
import random
import matplotlib.pyplot as plt

class KMeansClass:

    def __init__(self, data:list, k:int ):
        self.inputData = data
        self.centerPoint = []
        self.useK = k
        self.cluster = []

    def fit(self, max_iterations=100, show_progress=False):
        """K-Means 알고리즘을 실행하고 수렴할 때까지 반복합니다."""
        print("Starting K-Means algorithm.")
        self.firstCircle()

        for i in range(max_iterations):
            old_centerPoint = list(self.centerPoint)
            self.findCircle()
            self.updateCenter()

            if show_progress:
                print(f"--- Iteration {i+1} Result ---")
                self.plot(old_centerPoint=old_centerPoint)

            if old_centerPoint == self.centerPoint:
                print(f"Converged! (Total iterations: {i+1})")
                break
        
        print("K-Means fitting complete.")

    def findCircle(self):
        # 메서드가 호출될 때마다 클러스터를 새로 초기화(리셋)합니다.
        self.cluster = [[] for _ in range(self.useK)]
        for p in self.inputData:
            dis_to_center = [self.cal_dis(p, centerP) for centerP in self.centerPoint]
            min_center_index = -1
            min_dis = float('inf')
            for i, dis in enumerate(dis_to_center):
                if dis < min_dis:
                    min_dis = dis
                    min_center_index = i
            if min_center_index != -1:
                self.cluster[min_center_index].append(p)

    def updateCenter(self):
        """각 클러스터의 평균을 계산하여 중심점을 업데이트합니다."""
        new_center_points = []
        for cluster in self.cluster:
            if not cluster:
                continue
            num_points = len(cluster)
            dims = len(cluster[0])
            sums = [0] * dims
            for p in cluster:
                for i in range(dims):
                    sums[i] += p[i]
            avg_p = [s / num_points for s in sums]
            new_center_points.append(tuple(avg_p))
        self.centerPoint = new_center_points

    def firstCircle(self):
        self.centerPoint = []
        tempDis_index = []
        random_index = random.randrange(len(self.inputData))
        start_p = self.inputData[random_index]
        for i, point in enumerate(self.inputData):
            distance = self.cal_dis(start_p, point)
            tempDis_index.append((distance, i))
        tempDis_index.sort(key=self.first_item_return, reverse=True)
        self.centerPoint.append(self.inputData[random_index])
        self.centerPoint.append(self.inputData[tempDis_index[0][1]])
        if self.useK > 2:
            for _ in range(self.useK - 2):
                min_distance = []
                for p in self.inputData:
                    dis_to_center = [self.cal_dis(p, centerP) for centerP in self.centerPoint]
                    min_dis_point = min(dis_to_center)
                    min_distance.append(min_dis_point)
                max_dis = -1
                next_center_index = -1
                for i, dis in enumerate(min_distance):
                    if dis > max_dis:
                        max_dis = dis
                        next_center_index = i
                new_center = self.inputData[next_center_index]
                self.centerPoint.append(new_center)

    def first_item_return(self, _tuple: tuple):
        """튜플의 첫 번째 요소를 반환하는 헬퍼 함수"""
        return _tuple[0]

    def cal_dis(self, p1: tuple, p2: tuple):
        """두 점 사이의 유클리드 거리를 계산하는 내부 함수"""
        dist_sq = 0
        for i in range(len(p1)):
            dist_sq += (p1[i] - p2[i]) ** 2
        return math.sqrt(dist_sq)

    def plot(self, old_centerPoint=None):
        """클러스터링 결과를 시각화합니다."""
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        fig, ax = plt.subplots(figsize=(10, 8))

        if self.inputData:
            x_coords = [p[0] for p in self.inputData]
            y_coords = [p[1] for p in self.inputData]
            max_x = max(x_coords)
            max_y = max(y_coords)
            ax.set_xlim(0, max_x * 1.2)
            ax.set_ylim(0, max_y * 1.2)

        for i, cluster in enumerate(self.cluster):
            if cluster:
                points_x = [p[0] for p in cluster]
                points_y = [p[1] for p in cluster]
                ax.scatter(points_x, points_y, c=colors[i % len(colors)], label=f'Cluster {i}')
        
        if old_centerPoint:
            old_center_x = [c[0] for c in old_centerPoint]
            old_center_y = [c[1] for c in old_centerPoint]
            ax.scatter(old_center_x, old_center_y, marker='x', s=200, c='gray', alpha=0.5, label='Old Centroids')
            num_centers_to_compare = min(len(old_centerPoint), len(self.centerPoint))
            for i in range(num_centers_to_compare):
                old_p = old_centerPoint[i]
                new_p = self.centerPoint[i]
                ax.arrow(old_p[0], old_p[1], new_p[0] - old_p[0], new_p[1] - old_p[1],
                          head_width=1, head_length=1, fc='k', ec='k', length_includes_head=True)
        
        center_x = [c[0] for c in self.centerPoint]
        center_y = [c[1] for c in self.centerPoint]
        ax.scatter(center_x, center_y, marker='x', s=200, c='black', label='Current Centroids')
        
        ax.set_title('K-Means Clustering Result')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()
        ax.grid(True)
        plt.show()

# --- 예제 실행을 위한 함수 ---
def run_kmeans_example():
    print("--- Running K-Means Example ---")
    # 2차원 데이터 200개 생성
    data = []
    for i in range(200):
        data.append((random.uniform(1, 100), random.uniform(1, 100)))

    # --- 사용자에게 K값 입력받기 ---
    while True:
        try:
            k = int(input(" K 값을 입력하시오: "))
            if k > 0:
                break
            else:
                print("정수값 입력.")
        except ValueError:
            print("정수값 입력.")
    # --------------------------------

    # 1. KMeans 객체 생성
    kmeans = KMeansClass(data, k)
    
    # 2. fit 메서드 호출로 알고리즘 실행 (단계별 시각화 활성화)
    kmeans.fit(show_progress=True)

    # 3. 최종 결과 시각화
    print("--- Final Result ---")
    kmeans.plot()

run_kmeans_example()
