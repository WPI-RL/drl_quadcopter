
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QuadrotorNeuralNetwork(nn.Module):

    def __init__(self, n_rel_x, n_rel_y, n_rel_z, n_actions):
        super(QuadrotorNeuralNetwork, self).__init__()
        # Separated layers for each relative position
        self.x_layer1 = nn.Linear(n_rel_x, 16)
        self.x_layer2 = nn.Linear(16, 16)
        self.y_layer1 = nn.Linear(n_rel_y, 8)
        self.y_layer2 = nn.Linear(8, 8)
        self.z_layer1 = nn.Linear(n_rel_z, 8)
        self.z_layer2 = nn.Linear(8, 8)

        # Seperated layers for each camera image
        self.camera_1_layer1 = nn.Conv2d(kernel_size=10, in_channels=2, out_channels=8, stride=2)
        torch.nn.init.xavier_uniform_(self.camera_1_layer1.weight)
        print(self.camera_1_layer1.weight)
        self.camera_1_layer2 = nn.Conv2d(kernel_size=6, in_channels=8, out_channels=16, stride=1)
        self.camera_1_layer3 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32, stride=1)
        self.camera_1_layer4 = nn.Linear(800, 64)
        self.camera_1_layer5 = nn.Linear(64, 64)

        # Join the position layers
        self.joint_layer1 = nn.Linear((16 + 8 + 8), 16)

        self.joint_layer2 = nn.Linear((64 + 16), 64)
        self.joined_layer1 = nn.Linear(64, 32)

        self.output_layer = nn.Linear(32, 18)
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
        # if depth_im.dim() == 2:
        if depth_im.dim() == 0:
            depth_im_1 = depth_im[0]
            depth_im_1 = depth_im_1.unsqueeze(0)

        else:
            depth_im_1 = depth_im[:,0]
        #     depth_im_2 = depth_im[1]
        #     depth_im_2 = depth_im_2.unsqueeze(0)
        #     depth_im_3 = depth_im[2]
        #     depth_im_3 = depth_im_3.unsqueeze(0)
        # elif depth_im.dim() == 3:
        # depth_im_1 = depth_im[:,0]

        if self.debug:
            print("depth_im_1: ", depth_im_1)
            print("depth_im_1.dim(): ", depth_im_1.dim())
            print("depth_im_1.shape: ", depth_im_1.shape)

        cam11 = F.relu(self.camera_1_layer1(depth_im_1))
        cam12 = F.relu(self.camera_1_layer2(cam11))
        cam13 = F.relu(self.camera_1_layer3(cam12))
        #need serilization next
        cam14 = F.relu(self.camera_1_layer4(cam13))
        cam15 = F.relu(self.camera_1_layer5(cam14))

        if self.debug:
            print("cam1: ", cam1)
            print("cam1.dim(): ", cam1.dim())

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
        joint_pos = F.relu(self.joint_layer1(joint_pos))
        joint_all = torch.cat((joint_pos, cam15), dim=1)
        joint_all = F.relu(self.joint_layer2(joint_all))
        return self.output_layer(joint_all)
