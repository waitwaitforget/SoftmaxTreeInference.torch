require 'nn';
require 'nnx';

local SmtInference = {}
torch.manualSeed(0)
root_id = 29
input_size=10
--hierarchy={[1]=torch.IntTensor{2,3},[2]=torch.IntTensor{4,5},[3]=torch.IntTensor{6,7},[7]=torch.IntTensor{8,9}}
hierarchy = {[29]=torch.IntTensor{30,1,2}, [1]=torch.IntTensor{3,4,5}, [2]=torch.IntTensor{6,7,8}, [3]=torch.IntTensor{9,10,11},[4]=torch.IntTensor{12,13,14}, [5]=torch.IntTensor{15,16,17},
[6]=torch.IntTensor{18,19,20}, [7]=torch.IntTensor{21,22,23},[8]=torch.IntTensor{24,25,26,27,28}}

smt=nn.SoftMaxTree(input_size,hierarchy,root_id)

input=torch.rand(5,10)
input[1] = torch.randn(1,10)*3
target = torch.IntTensor{20,24,27,10,12}
pred=smt:forward{input,target}  --likelihood of each target class

cri=nn.TreeNLLCriterion()

loss=cri:forward(pred,target)


function smt_inference(smt,input,target)
	local sm = nn.SoftMax()
	local leaf_node = {}
	for i=1,smt.nChildNode do
		if not torch.any(smt.parentIds:eq(smt.childIds[i])) then
			leaf_node[#leaf_node+1] = smt.childIds[i]
		end
	end
	-- do a tranverse of the tree to compute the probability of each path
	function preorder_tranverse(pid)
	    if(probs[pid]==1) then
	        -- if it's a leaf node compute the probability and return
	        if not torch.any(smt.parentIds:eq(pid)) then
	            probs[pid]=probs[smt.childParent[pid][1]]*p[pid]
	            return
	        else -- get it's childids
	            local childIds = smt.childIds:narrow(1,smt.parentChildren[pid][1],smt.parentChildren[pid][2])
	            preact = preacts:narrow(1,smt.parentChildren[pid][1],smt.parentChildren[pid][2])
	            act = sm:forward(preact)

	            for j=1,act:nElement() do
	                p[childIds[j]] = act[j]
	            end
	           
	            local childp = smt.childParent[pid][1]
	            if childp == -1 then -- check if a root node
	                childp = smt.rootId
	            end
	            probs[pid]=probs[childp]*p[pid]
	            -- reverse the other nodes
	            for i=1,smt.parentChildren[pid][2] do
	                preorder_tranverse(childIds[i])
	            end
	        end
	    end
	end  
	--local pred=smt:forward{input,target}
	local pred = torch.IntTensor(target:size())
	for sample = 1,input:size(1) do
		preacts = torch.mv(smt.weight,input[sample]) + smt.bias
		p = torch.zeros(smt.bias:nElement()+1)
		p[smt.rootId]=1
		probs = torch.ones(smt.bias:nElement()+1)
		preorder_tranverse(smt.rootId)
		local pdist = probs:index(1,torch.LongTensor(leaf_node))
		local _,pt = torch.max(pdist,1)
		pred[sample] = leaf_node[pt[1]]
		print('input '..sample..' pred: '..(pred[sample])..' target: '..target[sample])
	end	
	collectgarbage()
end

--smt_inference(smt,input,target)
test = require('smtBatch')
--test = require('SmtInference')

pred,output = test.PharseOutput(smt,input,target)

--print(pred,output[1])