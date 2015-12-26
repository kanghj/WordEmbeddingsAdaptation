chromRegion = {};
gm12878 = {};
h1hesc = {};
helas3 = {};
hepg2 = {};
huvec = {};
k562 = {};
dataset = {};
dnaseIsignals = {}

i = 0;
CellTypesNumber = 6

print('\n-- dataset building --');
-- data building
file='data.csv'
local file1 = io.open(file)
for line in file1:lines() do
  if i >= 1 then
     chromRegion[i-1], gm12878[i-1], h1hesc[i-1], helas3[i-1], hepg2[i-1], huvec[i-1], k562[i-1] = unpack(line:split(","))
     profile = torch.Tensor(CellTypesNumber);
     profile = {tonumber(gm12878[i-1]), tonumber(h1hesc[i-1]), tonumber(helas3[i-1]), tonumber(hepg2[i-1]), tonumber(huvec[i-1]), tonumber(k562[i-1])}
     dnaseIsignals[i-1] = profile;
    profile = null;
  end
i = i + 1
end

chromRegionsNumber = #chromRegion+1
print('chromRegionsNumber '..chromRegionsNumber);

print(dnaseIsignals[0])
print(dnaseIsignals[chromRegionsNumber-1])

for i=1,chromRegionsNumber do   -- 15 times

   dataset[i] = {torch.Tensor(dnaseIsignals[i-1]), torch.Tensor(dnaseIsignals[i-1])} -- CHANGED -sc

   print(dataset[i]);
end
function dataset:size() return chromRegionsNumber end -- input dataset hgas
function dataset:dim() return chromRegionsNumber end -- input dataset hgas

inputs=dataset[1][1]:size(1)  -- CHANGED -sc
outputs=inputs; --
HUs = 5;
-- Creation of the neural network
print('-- creation of the neural network --');
require "nn"
mlp=nn.Sequential();  -- make a multi-layer perceptron
mlp:add(nn.Linear(inputs,HUs)) -- adds a (input x HUs)  layer to the network
mlp:add(nn.Sigmoid());
mlp:add(nn.Linear(HUs,outputs)) -- adds a (HUs x outputs)  layer to the network
mlp:add(nn.Sigmoid()); -- Sigmoid also in output, Peter says


-- Training
print('-- training --');
criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
print('dataset contains stuff like...')
print(dataset[1])
trainer:train(dataset) --> THE PROBLEMS START HERE! :-(

-- Testing
print('-- testing --');
geneProfileChosen = 0;
x=torch.Tensor(inputs);   -- create a test example Tensor. Test element: first gene
