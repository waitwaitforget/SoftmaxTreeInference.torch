
local SmtInference = {}

function SmtInference.PharseOutput(smt,input,target)
	local sm = nn.SoftMax()
	local leaf_node = {}
	for i=1,smt.nChildNode do
		if not torch.any(smt.parentIds:eq(smt.childIds[i])) then
			leaf_node[#leaf_node+1] = smt.childIds[i]
		end
	end
	-- do a tranverse of the tree to compute the probability of each path
	local function preorder_tranverse(pid)
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
	--local preacts = torch.Tensor()
	--local p = torch.Tensor()
	--local probs = torch.Tensor()
	local output = {}
	for sample = 1,input:size(1) do
		preacts = torch.mv(smt.weight,input[sample]) + smt.bias

		p = torch.zeros(smt.bias:nElement()+1)
		p[smt.rootId]=1
		probs = torch.ones(smt.bias:nElement()+1)
		preorder_tranverse(smt.rootId)
		local pdist = probs:index(1,torch.LongTensor(leaf_node))
		output[sample] = p
		local _,pt = torch.max(pdist,1)
		pred[sample] = leaf_node[pt[1]]
	end	
	
	collectgarbage()
	return pred,output
end

return SmtInference