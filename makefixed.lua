require('nn')
require('image')
require('loadcaffe')

torch.setdefaulttensortype('torch.FloatTensor')
local cnn = loadcaffe.load('vgg/VGG_ILSVRC_16_layers_deploy.prototxt', 'vgg/VGG_ILSVRC_16_layers.caffemodel', 'nn')

print(cnn)

local featureCNN = nn.Sequential()
for i = 1,17 do
    featureCNN:add( cnn:get(i) )
end
featureCNN:evaluate()
torch.save('./fixedCNN.t7', featureCNN);

