require 'torch'
local cjson = require "cjson"

local loadDataInfo = function(infoFile) 
    local allSamples = {}
    local fh = io.open(infoFile)
    while true do
        local line = fh:read()
        if line == nil then break end
        --print(line) 
        local info = cjson.decode(line)
        table.insert(allSamples, info)
    end
   
    local seq = torch.randperm( #allSamples )
    local splitNumber = math.floor(#allSamples * 0.85)
    local trainSamples = {}
    local verifySamples = {}

    for i = 1, splitNumber do
        table.insert(trainSamples, allSamples[seq[i]])
    end

    for i = splitNumber + 1, #allSamples do
        table.insert(verifySamples, allSamples[seq[i]])
    end
    
    return {trainSamples, verifySamples}
end

local db = loadDataInfo('./allSamples.list')
torch.save('allSamples.t7', db)

