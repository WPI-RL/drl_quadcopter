
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QuadrotorNeuralNetwork(nn.Module):

    def __init__(self, n_rel_x, n_rel_y, n_rel_z, n_actions):
        super(QuadrotorNeuralNetwork, self).__init__()
        # Separated layers for each relative position
        self.x_layer1 = nn.Linear(n_rel_x, 8)
        self.x_layer2 = nn.Linear(8, 8)
        self.y_layer1 = nn.Linear(n_rel_y, 8)
        self.y_layer2 = nn.Linear(8, 4)
        self.z_layer1 = nn.Linear(n_rel_z, 8)
        self.z_layer2 = nn.Linear(8, 4)

        # Seperated layers for each camera image
        self.camera_1_layer1 = nn.Linear(1024, 128)
        torch.nn.init.xavier_uniform_(self.camera_1_layer1.weight)
        print(self.camera_1_layer1.weight)

        self.camera_1_layer2 = nn.Linear(128, 16)
        self.camera_1_layer3 = nn.Linear(16, 8)
        self.camera_2_layer1 = nn.Linear(1024, 128)
        torch.nn.init.xavier_uniform_(self.camera_2_layer1.weight)
        self.camera_2_layer2 = nn.Linear(128, 16)
        self.camera_2_layer3 = nn.Linear(16, 8)
        self.camera_3_layer1 = nn.Linear(1024, 128)
        torch.nn.init.xavier_uniform_(self.camera_3_layer1.weight)
        self.camera_3_layer2 = nn.Linear(128, 16)
        self.camera_3_layer3 = nn.Linear(16, 16)

        # Join the position layers
        self.joint_layer1 = nn.Linear((8 + 4 + 4), 8)

        # Join the camera layers
        self.camera_joint_layer1 = nn.Linear((8 + 8 + 16), 16)

        self.joint_layer2 = nn.Linear((8 + 16), 32)
        self.output_layer = nn.Linear(32, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=.003)
        self.loss = nn.SmoothL1Loss()
        # Debug
        self.debug = False
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state):
        # Feed into camera layers
        depth_im = state.depth_image
        if depth_im.dim() == 2:
            depth_im_1 = depth_im[0]
            depth_im_1 = depth_im_1.unsqueeze(0)
            depth_im_2 = depth_im[1]
            depth_im_2 = depth_im_2.unsqueeze(0)
            depth_im_3 = depth_im[2]
            depth_im_3 = depth_im_3.unsqueeze(0)
        elif depth_im.dim() == 3:
            depth_im_1 = depth_im[:,0]
            depth_im_2 = depth_im[:,1]
            depth_im_3 = depth_im[:,2]

        if self.debug:
            print("depth_im_1: ", depth_im_1)
            print("depth_im_2: ", depth_im_2)
            print("depth_im_3: ", depth_im_3)
            print("depth_im_1.dim(): ", depth_im_1.dim())
            print("depth_im_2.dim(): ", depth_im_2.dim())
            print("depth_im_3.dim(): ", depth_im_3.dim())
            print("depth_im_1.shape: ", depth_im_1.shape)
            print("depth_im_2.shape: ", depth_im_2.shape)
            print("depth_im_3.shape: ", depth_im_3.shape)

        cam1 = F.relu(self.camera_1_layer1(depth_im_1))
        cam1 = F.relu(self.camera_1_layer2(cam1))
        cam1 = F.relu(self.camera_1_layer3(cam1))
        cam2 = F.relu(self.camera_2_layer1(depth_im_2))
        cam2 = F.relu(self.camera_2_layer2(cam2))
        cam2 = F.relu(self.camera_2_layer3(cam2))
        cam3 = F.relu(self.camera_3_layer1(depth_im_3))
        cam3 = F.relu(self.camera_3_layer2(cam3))
        cam3 = F.relu(self.camera_3_layer3(cam3))

        if self.debug:
            print("cam1: ", cam1)
            print("cam2: ", cam2)
            print("cam3: ", cam3)
            print("cam1.dim(): ", cam1.dim())
            print("cam2.dim(): ", cam2.dim())
            print("cam3.dim(): ", cam3.dim())

        # Feed into position layers
        rel_pos = state.relative_position
        if rel_pos.dim() == 1:
            rel_pos = rel_pos.unsqueeze(0)
        if rel_pos.dim() == 3:
            rel_pos = rel_pos.squeeze(1)
        rel_x = rel_pos[:,0]
        rel_x = rel_x.unsqueeze(1)
        rel_y = rel_pos[:,1]
        rel_y = rel_y.unsqueeze(1)
        rel_z = rel_pos[:,2]
        rel_z = rel_z.unsqueeze(1)
        if self.debug:
            print("rel_x: ", rel_x)
            print("rel_y: ", rel_y)
            print("rel_z: ", rel_z)
        x = F.relu(self.x_layer1(rel_x))
        x = F.relu(self.x_layer2(x))
        y = F.relu(self.y_layer1(rel_y))
        y = F.relu(self.y_layer2(y))
        z = F.relu(self.z_layer1(rel_z))
        z = F.relu(self.z_layer2(z))
        if self.debug:
            print("x: ", x)
            print("y: ", y)
            print("z: ", z)
            print("x.dim(): ", x.dim())
            print("y.dim(): ", y.dim())
            print("z.dim(): ", z.dim())
            print("x.shape: ", x.shape)
            print("y.shape: ", y.shape)
            print("z.shape: ", z.shape)

        joint_pos = torch.cat((x, y, z), dim=1)
        joint_cam = torch.cat((cam1, cam2, cam3), dim=1)
        joint_pos = F.relu(self.joint_layer1(joint_pos))
        joint_cam = F.relu(self.camera_joint_layer1(joint_cam))
        joint_all = torch.cat((joint_pos, joint_cam), dim=1)
        joint_all = F.relu(self.joint_layer2(joint_all))
        return self.output_layer(joint_all)
