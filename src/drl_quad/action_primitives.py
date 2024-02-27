import bezier
import numpy as np

class QuadActionPrimitives:
    def __init__(self, debug=False):
        # Set up debug flag
        self.debug = debug
        # Set up action primitives
        self.NUM_ACTIONS = 18
        self.actions = []
        # Set up rotation matrix for -45 degrees around the +ve x-axis
        self.x_rot_3d_neg45 = np.matrix(
            [[1.0, 0.0, 0.0],
            [0.0, np.cos(np.deg2rad(-45)), -np.sin(np.deg2rad(-45))],
            [0.0, np.sin(np.deg2rad(-45)), np.cos(np.deg2rad(-45))]],
            dtype=np.float32
        )
        # Set up action primitives
        self.setup_primitives()
    
    def convert_matrices_to_bezier(self, matrices, degree):
        # Convert to bezier curve
        for i in range(0, len(matrices)):
            # Convert to fortran array
            matrices[i] = np.asfortranarray(matrices[i])
            curve = bezier.Curve(matrices[i], degree=degree)
            self.actions.append(curve)

    def straight_line_rotations(self):
        # Manually generate the upward straight line, then rotate it
        # -45 degrees around the +ve x-axis 7 times
        matrix = np.zeros((8, 3, 2))
        # Force resulting diagonal lines to reach 1.0
        matrix[0] = np.matrix(
            [[0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.425]]
        )
        if self.debug:
            print("matrix1")
            print(matrix[0])
        for i in range(1, 8):
            matrix[i] = self.x_rot_3d_neg45 * matrix[i-1]
            if self.debug:
                print("matrix" + str(i+1))
                print(matrix[i])
        # Convert to bezier curve
        # Clip resulting matrices at -1 and +1 to match paper
        matrix[matrix<-1]=-1
        matrix[matrix>1]=1
        self.convert_matrices_to_bezier(matrix, degree=1)
    
    def curved_line_rotations(self):
        # Manually generate the upward curved line, then rotate it
        # -45 degrees around the +ve x-axis 7 times
        matrix = np.zeros((8, 3, 4))
        # Force resulting diagonal lines to reach 1.0 and -1.0
        matrix[0] = np.matrix(
            [[0.0, 0.5, 0.5, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.45, 1.45]],
        )
        if self.debug:
            print("matrix10")
            print(matrix[0])
        for i in range(1, 8):
            matrix[i] = self.x_rot_3d_neg45 * matrix[i-1]
            if self.debug:
                print("matrix" + str(i+10))
                print(matrix[i])
        # Convert to bezier curve
        # Clip resulting matrices at -1 and +1 to match paper
        matrix[matrix<-1]=-1
        matrix[matrix>1]=1
        self.convert_matrices_to_bezier(matrix, degree=3)

    def generate_ninth_action(self):
        """Generate a straight line along the +ve x-axis"""
        matrix = np.zeros((1, 3, 2))
        matrix[0] = np.matrix(
            [[0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0]]
        )
        if self.debug:
            print("matrix9")
            print(matrix[0])
        # Convert to bezier curve
        self.convert_matrices_to_bezier(matrix, degree=1)
    
    def setup_primitives(self):
        # Action 0 (just stay still)
        nodes_0 = np.asfortranarray([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ])
        curve_0 = bezier.Curve(nodes_0, degree=1)
        self.actions.append(curve_0)
        if self.debug:
            print("matrix0")
            print(nodes_0)

        # Generate straight line actions (1-8)
        self.straight_line_rotations()

        # Generate 9th action, which is a straight line along x-axis
        self.generate_ninth_action()

        # Generate curved line actions (10-17)
        self.curved_line_rotations()

    def get_sampled_curve(self, action, num_samples=5):
        # Get curve
        curve = self.actions[action]
        # Sample curve
        s_vals = np.linspace(0.0, 1.0, num_samples)
        points = curve.evaluate_multi(s_vals)
        return points

if __name__ == "__main__":
    # Test
    qap = QuadActionPrimitives(debug=False)
    # Choose an action at random
    for action in range(qap.NUM_ACTIONS):
        #print("Picking an action at random")
        #action = np.random.randint(0, qap.NUM_ACTIONS)
        print("action: " + str(action))
        # Get sampled curve
        points = qap.get_sampled_curve(action)
        print("points:")
        print(points)
        # Print last column
        print("points[:,4]")
        print(points[:,4])
