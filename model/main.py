from .att import *

class AttentionBlock(nn.Module):
	__constants__ = ['downsample']
	
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(AttentionBlock, self).__init__()
		norm_layer = nn.BatchNorm2d
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride
		self.ca = CoordAttention(in_channels=planes, out_channels=planes)

	def forward(self, x):
		identity = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		
		out = self.ca(out)
		
		if self.downsample is not None:
			identity = self.downsample(x)
		
		out += identity
		out = self.relu(out)
		
		return out

class fdn(nn.Module):
	
	def __init__(self, inplanes, planes):
		super(fdn, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(inplanes, planes)
		)
		self.relu = nn.ReLU(inplace=True)
		self.fc1 = nn.Sequential(
			nn.Linear(128, planes)
		)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):
		x1 = self.fc(x)
		x1 = self.relu(x1)
		x2 = self.fc1(x1)
		x2 = self.sigmoid(x2)
		out = x1*x2
		
		return out

class rarn(nn.Module):
	
	def __init__(self, block_b, block_a, layers, num_head=9, num_classes=99892):
		super(rarn, self).__init__()
		norm_layer = nn.BatchNorm2d
		self.num_head = num_head
		self._norm_layer = norm_layer
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = norm_layer(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		
		self.layer1 = self._make_layer(block_b, 64, 64, layers[0])
		self.layer2 = self._make_layer(block_b, 64, 128, layers[1], stride=2)
		self.fc = nn.Linear(128, 128)
		
		# In this branch, each BasicBlock replaced by AttentiveBlock.
		self.layer3_p1 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
		self.layer4_p1 = self._make_layer(block_a, 256, 512, layers[3], stride=1)
		for i in range(num_head):
			setattr(self, "latent_class_1_%d" % i, nn.Sequential(fdn(512, 128)))
		
		self.layer3_p2 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
		self.layer4_p2 = self._make_layer(block_a, 256, 512, layers[3], stride=1)
		for i in range(num_head):
			setattr(self, "latent_class_2_%d" % i, nn.Sequential(fdn(512, 128)))
		
		self.layer3_p3 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
		self.layer4_p3 = self._make_layer(block_a, 256, 512, layers[3], stride=1)
		for i in range(num_head):
			setattr(self, "latent_class_3_%d" % i, nn.Sequential(fdn(512, 128)))
		
		self.layer3_p4 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
		self.layer4_p4 = self._make_layer(block_a, 256, 512, layers[3], stride=1)
		for i in range(num_head):
			setattr(self, "latent_class_4_%d" % i, nn.Sequential(fdn(512, 128)))
		
		self.layer3_p5 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
		self.layer4_p5 = self._make_layer(block_a, 256, 512, layers[3], stride=1)
		for i in range(num_head):
			setattr(self, "latent_class_5_%d" % i, nn.Sequential(fdn(512, 128)))

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc_1 = nn.Linear(128, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	
	def _make_layer(self, block, inplanes, planes, blocks, stride=1):
		norm_layer = self._norm_layer
		downsample = None
		if stride != 1 or inplanes != planes:
			downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
		layers = []
		layers.append(block(inplanes, planes, stride, downsample))
		inplanes = planes
		for _ in range(1, blocks):
			layers.append(block(inplanes, planes))
		return nn.Sequential(*layers)
	
	def _forward_impl(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		
		x = self.layer1(x)
		x = self.layer2(x)

		patch_11 = x[:, :, 0:14, 0:14]
		patch_12 = x[:, :, 0:14, 14:28]
		patch_21 = x[:, :, 14:28, 0:14]
		patch_22 = x[:, :, 14:28, 14:28]
		patch_33 = x[:, :, 7:21, 7:21]
		
		branch_1_p1_out = self.layer3_1_p1(patch_11)
		branch_1_p1_out = self.layer4_1_p1(branch_1_p1_out)
		branch_1_p1_out = self.avgpool(branch_1_p1_out)
		branch_1_p1_out = torch.flatten(branch_1_p1_out, 1)
		heads_1 = []
		for i in range(self.num_head):
			heads_1.append(getattr(self, "latent_class_1_%d" % i)(branch_1_p1_out))
		head1 = torch.stack(heads_1)
		
		branch_1_p2_out = self.layer3_1_p2(patch_12)
		branch_1_p2_out = self.layer4_1_p2(branch_1_p2_out)
		branch_1_p2_out = self.avgpool(branch_1_p2_out)
		branch_1_p2_out = torch.flatten(branch_1_p2_out, 1)
		heads_2 = []
		for i in range(self.num_head):
			heads_2.append(getattr(self, "latent_class_2_%d" % i)(branch_1_p2_out))
		head2 = torch.stack(heads_2)
		
		branch_1_p3_out = self.layer3_1_p3(patch_21)
		branch_1_p3_out = self.layer4_1_p3(branch_1_p3_out)
		branch_1_p3_out = self.avgpool(branch_1_p3_out)
		branch_1_p3_out = torch.flatten(branch_1_p3_out, 1)
		heads_3 = []
		for i in range(self.num_head):
			heads_3.append(getattr(self, "latent_class_3_%d" % i)(branch_1_p3_out))
		head3 = torch.stack(heads_3)
		
		branch_1_p4_out = self.layer3_1_p4(patch_22)
		branch_1_p4_out = self.layer4_1_p4(branch_1_p4_out)
		branch_1_p4_out = self.avgpool(branch_1_p4_out)
		branch_1_p4_out = torch.flatten(branch_1_p4_out, 1)
		heads_4 = []
		for i in range(self.num_head):
			heads_4.append(getattr(self, "latent_class_4_%d" % i)(branch_1_p4_out))
		head4 = torch.stack(heads_4)
		
		branch_1_p5_out = self.layer3_1_p5(patch_33)
		branch_1_p5_out = self.layer4_1_p5(branch_1_p5_out)
		branch_1_p5_out = self.avgpool(branch_1_p5_out)
		branch_1_p5_out = torch.flatten(branch_1_p5_out, 1)
		heads_5 = []
		for i in range(self.num_head):
			heads_5.append(getattr(self, "latent_class_5_%d" % i)(branch_1_p5_out))
		head5 = torch.stack(heads_5)
		
		heads = head1 + head2 + head3 + head4 + head5
		# heads = torch.stack(heads).permute([1, 0, 2])
		heads = heads.sum(dim=0)
		out = self.fc_1(heads)
		
		return out, heads
	
	def forward(self, x):
		return self._forward_impl(x)

def RARN():
	return rarn(block_b=BasicBlock, block_a=AttentionBlock, layers=[2, 2, 2, 2])