import math
from collections import Counter
import matplotlib.pyplot as plt  
import random

class KMeansClass:

    def __init__(self, data:list, k:int ):
        self.inputData = data
        self.centerPoint = []
        self.useK = k
        self.cluster = []

    def findCircle(self):
        self.cluster = [ [] for _ in range(self.useK)]
        for p in self.inputData:

            dis_to_center = [self.cal_dis(p,centerP) for centerP in self.centerPoint]

            min_center_index = -1
            min_dis = float('inf')

            for i, dis in enumerate(dis_to_center):
                if dis < min_dis:
                    min_dis = dis
                    min_center_index = i               
            
            if min_center_index != -1:
                self.cluster[min_center_index].append(p)

    def fit(self, max_iterations=1000, show_progress=False):
        """K-Means 알고리즘을 실행하고 수렴할 때까지 반복합니다."""
        print("K-Means 알고리즘을 시작합니다.")
        self.firstCircle()

        for i in range(max_iterations):
            old_centerPoint = list(self.centerPoint)

            self.findCircle()
            self.updateCenter()

            # show_progress가 True일 경우, 매 단계마다 plot을 호출
            if show_progress:
                print(f"--- {i+1}번째 반복 결과 ---")
                self.plot(old_centerPoint=old_centerPoint) # 이전 중심점 정보를 전달

            if old_centerPoint == self.centerPoint:
                print(f"수렴 완료! (총 반복 횟수: {i+1})")
                break
        
        print("K-Means fitting 완료.")


    def updateCenter(self):

        new_centerP = []

        for cluster in self.cluster:
            if not cluster:
                continue

            num_points = len(cluster)
            dims = len(cluster[0])

            sums = [0] * dims

            for p in cluster:

                for i in range(dims):
                    sums[i] += p[i]
            
            avg_p = [ s / num_points for s in sums]

            new_centerP.append(tuple(avg_p))
        
        self.centerPoint = new_centerP



    def firstCircle(self):
        self.centerPoint = []
        tempDis_index = []

        random_index = random.randrange(0, len(self.inputData))


        start_p = self.inputData[random_index]

        for i, point in enumerate(self.inputData):

            distance = self.cal_dis(start_p,point)

            tempDis_index.append((distance,i))

        tempDis_index.sort(key = self.first_item_return,reverse=True)

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

                for i , dis in enumerate(min_distance):
                    if dis > max_dis:
                        max_dis = dis
                        next_center_index = i
                
                new_center = self.inputData[next_center_index]
                self.centerPoint.append(new_center)
            


    def first_item_return(self, _tuple:tuple):
        return _tuple[0]    
    
    def cal_dis(self, p1:tuple, p2:tuple):

        dist_sq = 0
        for i in range(len(p1)):
            dist_sq += (p1[i] - p2[i])**2
        return math.sqrt(dist_sq)

    def plot(self, old_centerPoint=None):
        """최종 클러스터링 결과를 시각화합니다."""
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        plt.clf() # 이전 그래프 내용을 지웁니다.
        
        # 1. 각 클러스터의 점들을 색상별로 그리기
        for i, cluster in enumerate(self.cluster):
            if cluster:
                points_x = [p[0] for p in cluster]
                points_y = [p[1] for p in cluster]
                plt.scatter(points_x, points_y, c=colors[i % len(colors)], label=f'Cluster {i}')

        # 2. 이전 중심점이 있으면 반투명하게 그리고 이동 경로 표시
        if old_centerPoint:
            old_center_x = [c[0] for c in old_centerPoint]
            old_center_y = [c[1] for c in old_centerPoint]
            plt.scatter(old_center_x, old_center_y, marker='x', s=200, c='gray', alpha=0.5, label='Old Centroids')

            # 이동 경로를 화살표로 그리기
            num_centers_to_compare = min(len(old_centerPoint), len(self.centerPoint))
            for i in range(num_centers_to_compare):
                old_p = old_centerPoint[i]
                new_p = self.centerPoint[i]
                plt.arrow(old_p[0], old_p[1], new_p[0] - old_p[0], new_p[1] - old_p[1],
                          head_width=1, head_length=1, fc='k', ec='k', length_includes_head=True)

        # 3. 현재 중심점들을 'X' 모양으로 그리기
        center_x = [c[0] for c in self.centerPoint]
        center_y = [c[1] for c in self.centerPoint]
        plt.scatter(center_x, center_y, marker='x', s=200, c='black', label='Current Centroids')

        plt.title('K-Means Clustering Result')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":

    data = []
    for i in range(200):
        data.append((random.uniform(1, 100), random.uniform(1, 100)))

    k = 3 # K 값 설정

    # 1. KMeans 객체 생성
    kmeans = KMeansClass(data, k)
    
    # 2. fit 메서드 호출로 알고리즘 실행 (매 단계 시각화 활성화)
    kmeans.fit(show_progress=True)

    # 3. 최종 결과 시각화
    print("--- 최종 결과 ---")
    kmeans.plot()
    

    