import numpy as np


def load_trajectory(path):
    data = np.load(path)
    positions = data['positions']
    orientations = data['orientations']
    return positions, orientations


def compare_trajectories(file1, file2):
    pos1, ori1 = load_trajectory(file1)
    pos2, ori2 = load_trajectory(file2)

    positions_equal = np.array_equal(pos1, pos2)
    orientations_equal = np.array_equal(ori1, ori2)

    print(f"Comparing: {file1} vs {file2}")
    print(f"Positions identical: {positions_equal}")
    print(f"Orientations identical: {orientations_equal}")

    if not positions_equal:
        print(f"  Position arrays differ. Shape1: {pos1.shape}, Shape2: {pos2.shape}")
        # Optionally, print where they differ
        if pos1.shape == pos2.shape:
            diff_idx = np.where(pos1 != pos2)
            print(f"  First differing index in positions: {diff_idx[0][0] if diff_idx[0].size > 0 else 'N/A'}")
    if not orientations_equal:
        print(f"  Orientation arrays differ. Shape1: {ori1.shape}, Shape2: {ori2.shape}")
        if ori1.shape == ori2.shape:
            diff_idx = np.where(ori1 != ori2)
            print(f"  First differing index in orientations: {diff_idx[0][0] if diff_idx[0].size > 0 else 'N/A'}")

    if positions_equal and orientations_equal:
        print("Trajectories are identical.")
    else:
        print("Trajectories are NOT identical.")


if __name__ == "__main__":
    # Hardcoded file paths
    file1 = 'data/cube_trajectory2.npz'
    file2 = 'data/cube_trajectory1.npz'
    compare_trajectories(file1, file2) 