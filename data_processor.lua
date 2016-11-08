require('torch')
require('image')

local boxSampling = require('boxsampling')
local classToNumber = require('data/classnumber')

local batchSize = 24 
local allShapes = { {256, 256} ,
                    {224, 288} ,
                    {288, 224} }

torch.setdefaulttensortype('torch.FloatTensor')
local dataProcessor = {}

local img2caffe = function(img)
    local mean_pixel = torch.Tensor({103.939, 116.779, 123.68})
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):mul(256.0)
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img:add(-1, mean_pixel)
    return img
end


dataProcessor._init = function(modelInfo)
    local self = dataProcessor
    self.modelInfo = modelInfo
    
    self.trainNumber = #self.trainSamples 
    self.trainPerm = torch.randperm(self.trainNumber)
    self.trainPos = 1

    self.verifyNumber = #self.verifySamples
    self.verifyPerm = torch.randperm(self.verifyNumber)
    self.verifyPos = 1
end

dataProcessor._buildTarget = function(targetWidth, targetHeight, labels, targetImg)
    local self = dataProcessor
    
    local _ = self.modelInfo.getSize(targetWidth, targetHeight)
    local lastWidth = _[1]
    local lastHeight = _[2]

    local targets = {}
    local masks = {}
    local predBoxes = boxSampling(self.modelInfo, targetWidth, targetHeight, labels)
    local pindex = 1    -- same order with boxSampling
    for i = 1, #self.modelInfo.boxes do
        local wid = lastWidth - (self.modelInfo.boxes[i][1] - 1)
        local hei = lastHeight - (self.modelInfo.boxes[i][2] - 1)

        local conf = torch.zeros(hei, wid)
        local loc = torch.zeros(4, hei, wid)
        local confMask = torch.zeros(21, hei, wid)
        local locMask = torch.zeros(4, hei, wid)
        
        for h = 1,hei do
            for w = 1,wid do
                local pbox = predBoxes[pindex]
                pindex = pindex + 1
                
                -- skkiped predition
                if ( pbox.label == -1) then
                    -- set to backgroud
                    conf[h][w] = self.modelInfo.classNumber
                end
                
                -- negative predition
                if ( pbox.label == 0) then
                    confMask[{{}, h, w}] = 1
                    
                    -- background
                    conf[h][w] = self.modelInfo.classNumber
                end
                
                -- positive predition
                if ( pbox.label > 0) then
                    confMask[{{}, h, w}] = 1
                    locMask[{{}, h, w}] = 1
                    
                    -- object
                    conf[h][w] = classToNumber( labels[pbox.label].class )
                   
                    -- location
                    loc[1][h][w] = (labels[pbox.label].xmin - pbox.xmin) / 16 
                    loc[2][h][w] = (labels[pbox.label].ymin - pbox.ymin) / 16
                    loc[3][h][w] = (labels[pbox.label].xmax - pbox.xmax) / 16
                    loc[4][h][w] = (labels[pbox.label].ymax - pbox.ymax) / 16
                    
                    --[[
                    targetImg = image.drawRect(targetImg, labels[pbox.label].xmin, labels[pbox.label].ymin, labels[pbox.label].xmax, labels[pbox.label].ymax);  
                    targetImg = image.drawRect(targetImg, pbox.xmin+2, pbox.ymin+2, pbox.xmax-2, pbox.ymax-2, {color = {0, 255, 0}});
                    --]]
                end
            end
        end

        table.insert(targets, conf)
        table.insert(targets, loc)
        table.insert(masks, confMask)
        table.insert(masks, locMask)
    end
    
    --[[
    local randFile = './images/' .. math.random() .. '.jpg'
    image.save(randFile, targetImg)
    --]]

    return targets, masks
end

dataProcessor.doSampling = function(isVerify)
    local self = dataProcessor
   
    local _ = math.floor(math.random() * 100) % 3 + 1
    local targetWidth = allShapes[_][1]
    local targetHeight = allShapes[_][2]

    local _ = self.modelInfo.getSize(targetWidth, targetHeight)
    local lastWidth = _[1]
    local lastHeight = _[2]

    local xinput = torch.Tensor(batchSize, 3, targetHeight, targetWidth)
    local targets = {}
    local masks = {}

    for i = 1, #self.modelInfo.boxes do
        local wid = lastWidth - (self.modelInfo.boxes[i][1] - 1)
        local hei = lastHeight - (self.modelInfo.boxes[i][2] - 1)

        local conf = torch.Tensor(batchSize, hei, wid)
        local loc = torch.Tensor(batchSize, 4, hei, wid)
        table.insert(targets, conf)
        table.insert(targets, loc)

        local confMask = torch.zeros(batchSize, 21, hei, wid)
        local locMask = torch.zeros(batchSize, 4, hei, wid)
        table.insert(masks, confMask)
        table.insert(masks, locMask)
    end

    local i = 1
    while ( i <= batchSize ) do
        local ii = self.trainPerm[self.trainPos] 
        local info = self.trainSamples[ii]
        if ( isVerify == true) then
            ii = self.verifyPerm[self.verifyPos] 
            info = self.verifySamples[ii]
        end

        local targetImg, labels = self._processImage(info, targetWidth, targetHeight)
 
        if ( targetImg ~= nil) then
            xinput[i]:copy( targetImg);
 
            local ts, ms = self._buildTarget(targetWidth, targetHeight, labels, targetImg)
            for j = 1, #ts do
                targets[j][i]:copy( ts[j] )
                masks[j][i]:copy( ms[j] )
            end

            i = i + 1
        end

        if ( isVerify == true ) then
            self.verifyPos = self.verifyPos + 1
            if ( self.verifyPos > self.verifyNumber) then
                self.verifyPos = 1
            end
        else
            self.trainPos = self.trainPos + 1
            if ( self.trainPos > self.trainNumber) then
                self.trainPos = 1
            end
        end
    end
    collectgarbage();

    return {xinput, targets, masks}
end

dataProcessor.doVerifySampling = function()
    local self = dataProcessor
    return self.doSampling(true)
end

-- image random sampling 
dataProcessor._processImage = function(info, targetWidth, targetHeight)
    local img = image.loadJPG('./data/' .. info['image']['file'])
    local wid = img:size()[3]
    local hei = img:size()[2]
   
    -- scale and crop
    local scale = targetWidth / wid
    local cutWid, cutHei = targetWidth , math.floor(hei * scale)
    local offsetx, offsety = 0, math.floor(math.random() * (cutHei - targetHeight) )
    if ( cutHei < targetHeight) then
        scale = targetHeight / hei
        cutWid, cutHei = math.floor(wid * scale), targetHeight
        offsety, offsetx = 0, math.floor(math.random() * (cutWid - targetWidth) )
    end

    if ( cutWid <= targetWidth ) then
        cutWid = targetWidth 
        offsetx = 0
    end
    if ( cutHei <= targetHeight ) then
        cutHei = targetHeight
        cutsety = 0;
    end
    
    local labels = {}
    local anns = info["annotation"]
    for i = 1, #anns do
        local bbox = {}
        bbox.class = anns[i]['category_id']
   
        bbox.xmin = math.floor(anns[i]['bbox'][1]*scale) - offsetx
        bbox.ymin = math.floor(anns[i]['bbox'][2]*scale) - offsety
        bbox.xmax = math.floor(anns[i]['bbox'][3]*scale) + bbox.xmin 
        bbox.ymax = math.floor(anns[i]['bbox'][4]*scale) + bbox.ymin
        
        if ( bbox.xmax > 0 and bbox.ymax > 0 and bbox.xmin < targetWidth and bbox.ymin < targetHeight ) then
            if ( bbox.xmin < 0) then
                bbox.xmin = 0
            end
            if ( bbox.ymin < 0) then
                bbox.ymin = 0
            end
            if ( bbox.xmax >= targetWidth ) then
                bbox.xmax = targetWidth
            end
            if ( bbox.ymax >= targetHeight ) then
                bbox.ymax = targetHeight
            end

            if ( (bbox.ymax - bbox.ymin) > 16 and (bbox.xmax - bbox.xmin) > 16 ) then            
                table.insert(labels, bbox)
            end
        end
    end

    if ( #labels == 0) then
        return nil
    end

    local scaledImg = image.scale(img, cutWid, cutHei)
    local targetImg = image.crop(scaledImg, offsetx, offsety, offsetx + targetWidth, offsety + targetHeight) 

    if ( math.random() > 0.5) then
        targetImg = image.hflip(targetImg)
        for i = 1, #labels do
            local temp = targetWidth - labels[i].xmin
            labels[i].xmin = targetWidth - labels[i].xmax
            labels[i].xmax = temp
        end
    end

    --[[
    for i = 1, #labels do
        local bbox = labels[i]
        targetImg = image.drawRect(targetImg, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax);  
    end
    --local randFile = './images/' .. math.random() .. '.jpg'
    --image.save(randFile, targetImg)
    --]]

    targetImg = img2caffe(targetImg)
    return targetImg, labels
end

dataProcessor.trainSamples = infoDB[1]
dataProcessor.verifySamples = infoDB[2]

return dataProcessor
