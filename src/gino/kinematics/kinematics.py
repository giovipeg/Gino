# ruff: noqa: N806

import os

import numpy as np

try:
    import pinocchio as pin
except ImportError:
    raise ImportError(
        "This feature requires `pinocchio`, which is not available via pip. "
        "Please install it manually with: `conda install -c conda-forge pinocchio`"
    )

class RobotKinematics:
    """
    Thin Pinocchio wrapper.
        • FK   -> fk(q, frame)
        • J    -> jacobian(q, frame)
        • IK   -> ik(q0, target_T, frame)

    Where:
        - q         : robot's current joint configuration
        - q0        : initial guess for joint angles
        - target_T  : target pose
    """
    def __init__(self, urdf_path: str, frame_name: str = "gripper_link"):
        if not urdf_path.endswith(".urdf"):
            urdf_path = os.path.join(os.path.dirname(__file__), "urdf", f"{urdf_path}.urdf")

        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(frame_name)
        self.frame_name = frame_name
        self.workspace_center, self.workspace_radius = self._calculate_workspace()
        print(f"Workspace center: {self.workspace_center}, radius: {self.workspace_radius}")

    def _calculate_workspace(self):
        # --- Workspace calculation (sum of link lengths) ---
        # The workspace radius is the sum of the distances between consecutive link origins
        # from shoulder_link to gripper_link in the neutral configuration.
        link_chain = [
            "shoulder_link",
            "humerus_link",
            "forearm_link",
            "wrist_link",
            "gripper_link",
        ]
        q_neutral = pin.neutral(self.model)
        positions = []
        for link in link_chain:
            pos = self.fk(q_neutral, link)[:3, 3]
            positions.append(pos)
        # Sum the Euclidean distances between consecutive link origins
        radius = sum(np.linalg.norm(positions[i+1] - positions[i]) for i in range(len(positions)-1))
        center = positions[0]  # shoulder_link origin
        return center, radius

    # ---------- Forward kinematics ----------
    def fk(self, q, frame: str | None = None):
        """Return 4×4 SE(3) of desired frame (default gripper_tip)."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        fid = self.model.getFrameId(frame or self.frame_name)
        if self.data.oMf is None:
            raise RuntimeError(f"Frame placement for frame id {fid} is not initialized. Ensure that forward kinematics has been computed correctly.")
        return self.data.oMf[fid].homogeneous

    # ---------- Jacobian ----------
    def jacobian(self, q, frame: str | None = None, reference_frame=None):
        """Return 6×N frame Jacobian."""
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        fid = self.model.getFrameId(frame or self.frame_name)
        if reference_frame is None:
            reference_frame = pin.LOCAL_WORLD_ALIGNED
        return pin.getFrameJacobian(self.model, self.data, fid, reference_frame)

    # ---------- Inverse kinematics (Gauss–Newton with damping) ----------
    def ik(
        self,
        q0,                         # starting guess (len == model.nq)
        target_t,                   # 4×4 desired pose
        frame: str | None = None,
        tol: float = 1e-3,
        max_iters: int = 5,
        damping: float = 1e-4,
    ):
        fid = self.model.getFrameId(frame or self.frame_name)
        q = np.array(q0, dtype=np.float64)

        for _ in range(max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            if self.data.oMf is None:
                raise RuntimeError("self.data.oMf is not initialized. Make sure Pinocchio is properly installed and the model/data are valid.")
            current_T = self.data.oMf[fid]

            target_SE3 = pin.SE3(target_t[:3, :3], target_t[:3, 3])
            err6 = pin.log6(current_T.inverse() * target_SE3)

            if np.linalg.norm(err6) < tol:
                return pin.normalize(self.model, q)

            J = self.jacobian(q, frame, pin.LOCAL)
            H = J.T @ J + damping * np.eye(J.shape[1])
            dq = np.linalg.solve(H, J.T @ err6)

            q[: len(dq)] += dq
            q = pin.normalize(self.model, q)

        return q  # Return best effort even if not converged

    def is_in_workspace(self, xyz, offset):
        """Check if a 3D point is inside the robot's spherical workspace."""
        xyz = np.asarray(xyz)
        return np.linalg.norm(xyz - self.workspace_center) <= (self.workspace_radius - offset)
