local cjson = require "cjson"

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local result = nil
local DataLoader = {}

function DataLoader.new(n, trainDBFile, modelInfo, randSeed)
  local self = {}
  for k,v in pairs(DataLoader) do
    self[k] = v
  end
  
  if ( randSeed == nil) then
    randSeed = 2014  
  end

  self.threads = Threads(n,
                         function(idx)
                           torch.manualSeed(randSeed + idx)
                           infoDB = torch.load(trainDBFile)
                           dataProcessor = paths.dofile('data_processor.lua')
                           dataProcessor._init(modelInfo)
                           -- cleaning global var
                           infoDB = nil
                         end)
   for i = 1, n do
     self.threads:addjob(self._getBatchFromThreads, self._pushResult)
   end 
   
   return self
end

function DataLoader._getBatchFromThreads()
    return dataProcessor.doSampling()  
end

function DataLoader._pushResult(...)
    result = ...
end

function DataLoader._getVerifyFromThreads()
    return dataProcessor.doVerifySampling()  
end

function DataLoader:getBatch()
   self.threads:addjob(self._getBatchFromThreads, self._pushResult)
   self.threads:dojob() 
   return result
end

function DataLoader:getVerifyBatch()
   self.threads:addjob(self._getVerifyFromThreads, self._pushResult)
   self.threads:dojob() 
   return result
end

return DataLoader
