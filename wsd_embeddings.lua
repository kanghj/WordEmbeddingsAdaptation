require 'torch'
require 'nn'
require 'paths'

-- embeddings from senna (c&w)
-- 1 for the context's word embeddings
-- 1 for pos tags
ltw = nn.LookupTable(130000, 50)
ltw2 = nn.LookupTable(46, 46)

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
end


trainSize = 5
testSize = 5

-- create training data set
for f in paths.files("./testtxt/testfiles/") do

   -- print p to see got wad
   if not paths.dirp("./testtxt/testfiles/" .. f) then

--      inputFile = torch.DiskFile("testtxt/testfiles/" .. f, 'r')
      local inputFile = io.open("testtxt/testfiles/" .. f, 'r')

      inputLine = torch.LongStorage(20)

      n_classes = tonumber(inputFile:read('*line'))

      -- init new model
      pt = nn.ParallelTable()
      pt:add(ltw)
      pt:add(ltw2)

      --- todo: add capitalization feature
      mlp = nn.Sequential()
      mlp:add(pt)
      --mlp:add(ltw)

      -- the NN layers
      jt = nn.JoinTable(2)
      bef_dropout_reshape = nn.Reshape(960)
      dropout = nn.Dropout(0.5)
      reshape = nn.Reshape(960)
      linear = nn.Linear(960, n_classes)
      lsm = nn.LogSoftMax()

      mlp:add(jt)
      mlp:add(bef_dropout_reshape)
      mlp:add(dropout)
      mlp:add(reshape)
      mlp:add(linear)
      mlp:add(lsm)
      dataset = {}

      function dataset:size() return trainSize end
      trainSize = 0
      while true  do
	 local instance_values = inputFile:read('*line')
	 if not instance_values then break end
--		 inputFile:readLong(inputLine)
	 inputLine = instance_values:split(' ')
	 trainSize = trainSize + 1
	 local input = torch.Tensor(20)
	 for j=1,10 do 
	    input[j] = tonumber(inputLine[j])
	 end
	 for j=11,20 do 
	    input[j] = tonumber(inputLine[j]) + 1
	 end

	 local label = tonumber(inputFile:read('*line'))
	 local newInput = nn.SplitTable(1):forward(nn.Reshape(2,10):forward(input))

	 local labelTensor = torch.Tensor(1)
	 labelTensor[1] = label
	 dataset[trainSize] = {newInput, labelTensor}
      end

      inputFile:close()

      criterion = nn.ClassNLLCriterion()
      trainer = nn.StochasticGradient(mlp, criterion)
      trainer.learningRate = 0.01
      trainer:train(dataset)
      
      print('done with ' .. f .. ', training size is ' .. trainSize)

   end
end

--outputEmbeddingsFile = torch.DiskFile('new_embeddings.txt', 'w')
outputEmbeddingsFile = io.open('new_embeddings.txt', 'w+')
--io.output(outputEmbeddingsFile)
for i=1, (#ltw.weight)[1] do
    for j=1, (#ltw.weight[i])[1] do
        outputEmbeddingsFile:write(ltw.weight[i][j], ' ')
    end
    outputEmbeddingsFile:write('\n')
end
outputEmbeddingsFile:close(outputEmbeddingsFile)


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

