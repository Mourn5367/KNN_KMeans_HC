import math
from collections import Counter
import matplotlib.pyplot as plt

class KnnClass:

    def __init__(self):
        self.sample_data = []
        self.labels = []
        self.k = 0
        self.new_point = []
        self.show_columns = ()
        self.dist_info_list = []

    def init_data(self, sample_data:list, labels:list, k:int, new_point:list, show_columns:tuple):
        self.sample_data = sample_data
        self.labels = labels
        self.k = k
        self.new_point = new_point
        self.show_columns = show_columns
        self.dist_info_list = [] # 재실행을 위해 초기화

        self.cal_dist()
        prediction = self.determine_knn()
        print(f"\nThe predicted class for the new data {self.new_point} is '{prediction}'.")
        
        if self.show_columns:
            self.visualize_knn()

    def cal_dist(self):
        """
        거리 계산 시 (거리, 원본 좌표, 레이블) 형태로 저장하도록 수정
        """
        for i, sample_point in enumerate(self.sample_data):
            dist = self.euclidean_distance(sample_point, self.new_point)
            self.dist_info_list.append((dist, sample_point, self.labels[i]))
        self.dist_info_list.sort(key=self.first_item_return)

    def determine_knn(self):
        # 상위 k개의 이웃 선택
        k_nearest_neighbors = self.dist_info_list[:self.k]
        # 이웃들의 레이블만 추출
        neighbor_labels = [item[2] for item in k_nearest_neighbors]
        # 다수결 투표
        most_common = Counter(neighbor_labels).most_common(1)
        return most_common[0][0]
    
    def visualize_knn(self):
        """
        show_columns에 지정된 피쳐를 기준으로 KNN 결과를 시각화합니다.
        """
        if not self.show_columns or len(self.show_columns) != 2:
            print("Two axes were not specified for visualization.")
            return

        x_index = self.show_columns[0] - 1
        y_index = self.show_columns[1] - 1

        fig, ax = plt.subplots(figsize=(10, 8))

        # 데이터의 범위를 계산하여 축의 한계를 0부터 최대값의 1.2배까지로 설정
        all_points = self.sample_data + [self.new_point]
        x_coords = [p[x_index] for p in all_points]
        y_coords = [p[y_index] for p in all_points]
        
        max_x = max(x_coords) if x_coords else 0
        max_y = max(y_coords) if y_coords else 0
        
        ax.set_xlim(0, max_x * 1.2)
        ax.set_ylim(0, max_y * 1.2)

        unique_labels = sorted(list(set(self.labels)))
        colors = plt.cm.rainbow([i/len(unique_labels) for i in range(len(unique_labels))])
        color_map = {label: color for label, color in zip(unique_labels, colors)}

        # 전체 샘플 데이터 그리기
        for label in unique_labels:
            points = [p for p, l in zip(self.sample_data, self.labels) if l == label]
            if points:
                ax.scatter([p[x_index] for p in points], [p[y_index] for p in points], color=color_map[label], label=label)

        # 새로운 데이터 포인트 그리기
        ax.scatter(self.new_point[x_index], self.new_point[y_index], color='black', marker='*', s=200, edgecolor='white', label='New Point')

        # K개의 이웃 강조하기
        k_nearest_neighbors = self.dist_info_list[:self.k]
        for i, (_, point, _) in enumerate(k_nearest_neighbors):
            label = 'k-Nearest' if i == 0 else ""
            ax.scatter(point[x_index], point[y_index], facecolors='none', edgecolors='green', s=150, linewidths=2, label=label)

        # 원 그리기
        if self.dist_info_list and self.k <= len(self.dist_info_list):
            radius = self.dist_info_list[self.k - 1][0]
            circle = plt.Circle(
                (self.new_point[x_index], self.new_point[y_index]),
                radius, facecolor='yellow', alpha=0.4, edgecolor='green', linestyle='-', linewidth=2.5
            )
            ax.add_patch(circle)

        plt.title(f'KNN Visualization (k={self.k})')
        plt.xlabel(f'Feature {x_index + 1}')
        plt.ylabel(f'Feature {y_index + 1}')
        
        ax.legend()
        
        plt.grid(True)
        plt.show()

    def euclidean_distance(self, sampleD:list, inputD:list):
        dist_sq = 0
        for i in range(len(sampleD)):
            dist_sq += (sampleD[i] - inputD[i])**2
        return math.sqrt(dist_sq)
    
    def first_item_return(self, _tuple:tuple):
        return _tuple[0]

def get_data_from_terminal():
    """터미널에서 사용자 입력을 받아 KNN 분석에 필요한 모든 데이터를 반환합니다."""
    while True:
        try:
            k = int(input("Enter the value for K: "))
            break
        except ValueError:
            print("Error: You must enter a number.")

    while True:
        try:
            raw_input_str = input("Enter the coordinates for the new data point (e.g., 5,11): ")
            new_point = [float(x) for x in raw_input_str.split(',')]
            break
        except (ValueError, IndexError):
            print("Error: Please enter in the format 'number,number'.")

    print("\nEnter training data (e.g., 4,12,C). Type 'done' to finish.")
    sample_data = []
    labels = []
    while True:
        raw_input_str = input(f"Data {len(sample_data) + 1}: ").strip()
        if raw_input_str.lower() in ['done', 'exit']:
            if not sample_data:
                print("Error: At least one training data point is required.")
                continue
            break
        try:
            parts = raw_input_str.split(',')
            if len(parts) < 2:
                raise ValueError("Incorrect input format.")
            coords = [float(x) for x in parts[:-1]]
            label = parts[-1].strip()
            sample_data.append(coords)
            labels.append(label)
        except (ValueError, IndexError):
            print("Error: Please enter in the format 'coord,coord,label' (e.g., 4,12,C).")
    return sample_data, labels, k, new_point

def get_visualization_columns(sample_data: list) -> tuple:
    """
    학습 데이터를 기반으로 피쳐 개수를 확인하고,
    사용자에게 시각화할 두 축을 입력받아 튜플로 반환합니다.
    """
    if not sample_data:
        return ()
    feature_count = len(sample_data[0])
    show_cols = ()
    if feature_count >= 2:
        print(f"\nThe data has {feature_count} features.")
        raw_cols = input("Enter the two axis numbers for visualization (e.g., 1,2 / press Enter to skip): ")
        try:
            if raw_cols:
                cols = tuple(int(c) for c in raw_cols.split(','))
                if len(cols) == 2:
                    show_cols = cols
                else:
                    print("Error: You must enter two axis numbers.")
        except ValueError:
            print("Error: Please enter numbers separated by a comma.")
    return show_cols

# --- 예제 실행을 위한 함수 ---

# def run_knn_predefined_example():
#     """미리 정의된 데이터로 KNN을 실행하는 예제"""
#     print("\n------------------------------------------")
#     print("--- Running KNN Example (Predefined Data) ---")
#     print("------------------------------------------")
    
#     sample_data = [[2, 4], [4, 6], [4, 8], [6, 4], [6, 6], [8, 2]]
#     labels = ['A', 'A', 'A', 'B', 'B', 'B']
#     k = 3
#     new_point = [5, 5]
#     show_cols = (1, 2)

#     print(f"K = {k}, New Point = {new_point}")
    
#     knn = KnnClass()
#     knn.init_data(sample_data, labels, k, new_point, show_cols)

def run_knn_interactive_example():
    """터미널 입력을 통해 KNN을 실행하는 예제"""
    print("\n------------------------------------------")
    print("--- Running KNN Example (User Input) ---")
    print("------------------------------------------")
    knn = KnnClass()
    sam, label, k_val, input_d = get_data_from_terminal()
    show_cols = get_visualization_columns(sam)
    knn.init_data(sam, label, k_val, input_d, show_cols)

# --- 코랩에서 직접 실행 ---
# 아래 함수 중 원하는 예제의 주석을 해제하거나, 별도의 셀에서 호출하여 실행하세요.

# run_knn_predefined_example()
run_knn_interactive_example() # 코랩에서는 텍스트 입력을 지원하는 셀에서만 동작합니다.
