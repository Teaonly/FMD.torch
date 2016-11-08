require('nn')
require('image')
require('loadcaffe')

-- Download VGG16 caffe model
-- https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

torch.setdefaulttensortype('torch.FloatTensor')
local cnn = loadcaffe.load('vgg/VGG_ILSVRC_16_layers_deploy.prototxt', 'vgg/VGG_ILSVRC_16_layers.caffemodel', 'nn')

print(cnn)

local featureCNN = nn.Sequential()
for i = 1,17 do
    featureCNN:add( cnn:get(i) )
end
featureCNN:evaluate()
torch.save('./fixedCNN.t7', featureCNN);

