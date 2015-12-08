require 'torch'
require 'nn'


-- embeddings from senna (c&w)
-- 1 for before target word
-- 1 for after target word
ltw = nn.LookupTable(130000, 50)
ltw2 = nn.LookupTable(130000, 50)

-- initialize lookup table with embeddings
embeddingsFile = torch.DiskFile('embeddings.txt');
embedding = torch.DoubleStorage(50)

-- todo : change hardcoded values like 130000
for i=1, 130000 do 
   embeddingsFile:readDouble(embedding);
   local emb = torch.Tensor(50)
   for j=1,50 do 
      emb[j] = embedding[j]
   end
   ltw.weight[i] = emb;
   ltw2.weight[i] = emb;
end

n_classes = 40

--pt = nn.ParallelTable()
--pt:add(ltw)

--- todo: add capitalization feature
mlp = nn.Sequential()
--mlp:add(pt)
mlp:add(ltw)

-- the NN layers
dropout = nn.Dropout(0.5)
reshape = nn.Reshape(500)
linear = nn.Linear(500, n_classes)
lsm = nn.LogSoftMax()

mlp:add(dropout)
mlp:add(reshape)
mlp:add(linear)
mlp:add(lsm)

trainSize = 5
testSize = 5
-- create training data set
inputFile = torch.DiskFile('wsd_training.txt', 'r')
inputLine = torch.LongStorage(10)

dataset = {}
function dataset:size() return trainSize end

for i=1,dataset:size()  do 
   inputFile:readLong(inputLine)
   local input = torch.Tensor(10)
   for j=1,10 do 
      input[j] = inputLine[j]
   end

   local label = inputFile:readInt()
   inputTensor = torch.Tensor(input)
   labelTensor = torch.Tensor(1)
   labelTensor[1] = label
   dataset[i] = {inputTensor, labelTensor}
   print(label)
end


inputFile:close()

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)


-- Testing
print('Testing....')

outputFile = torch.DiskFile('wsd_testing_results.txt', 'w')

testFile = torch.DiskFile('wsd_testing.txt', 'r')
inputLine = torch.IntStorage(10)

for i=1,testSize do 
   testFile:readInt(inputLine)
   local input = torch.Tensor(10)
   for j=1,10 do 
      input[j] = inputLine[j]
   end
   local label = testFile:readInt()
   inputTensor = torch.Tensor(input)
   
   output = mlp:forward(inputTensor)

   local outputLabel = 1;
   local outputValue = -1000;

   for k=1, output:size()[1] do
      --print(k .. ', '  .. output[k])
      if output[k] > outputValue then
	 outputLabel = k;
	 outputValue = output[k];
      end
   end

   outputFile:writeInt(outputLabel);
end

testFile:close()
outputFile:close()

