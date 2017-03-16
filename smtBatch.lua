
local SmtInference = {}

function SmtInference.PharseOutput(smt,input,target)
	local sm = nn.SoftMax()
	--sm:cuda()
	local leaf_node = {}

	for i=1,smt.nChildNode do
		if not torch.any(smt.parentIds:eq(smt.childIds[i])) then
			leaf_node[#leaf_node+1] = smt.childIds[i]
		end
	end
	-- define some variables 
	local nSample = input:size(1)
	local pred = torch.IntTensor(target:size())
	
	local preacts = torch.add(torch.mm(input,smt.weight:t()),torch.repeatTensor(smt.bias,nSample,1))
	-- check ok
	preacts = preacts:typeAs(input)
	
	local p = torch.zeros(nSample, smt.nChildNode+1)
	local probs = torch.ones(nSample, smt.nChildNode+1)
	p:select(2,smt.rootId):fill(1)
	-- do a tranverse of the tree to compute the probability of each path
	local function preorder_tranverse(pid)
	    if(probs[1][pid]==1) then
	        -- if it's a leaf node compute the probability and return
	        if not torch.any(smt.parentIds:eq(pid)) then
	        	
	        	local tmp = probs:select(2,smt.childParent[pid][1]):clone()
	            probs:select(2,pid):copy(tmp:cmul(p:select(2,pid)))
	            
	            return
	        else -- get it's childids
	            local childIds = smt.childIds:narrow(1,smt.parentChildren[pid][1],smt.parentChildren[pid][2])
	           
	            local preact = preacts:narrow(2,smt.parentChildren[pid][1],smt.parentChildren[pid][2])
	            local act = sm:forward(preact)

	            for j=1 , act:size(2) do
	                p:select(2,childIds[j]):copy(act:select(2,j))
	            end
				if pid == smt.rootId then childp = smt.rootId 
	            else childp = smt.childParent[pid][1] end
	            if childp == -1 then -- check if a root node
	                childp = smt.rootId
	            end
	            local tmp = probs:select(2,childp):clone()
	            probs:select(2,pid):copy(tmp:cmul(p:select(2,pid)))
	            -- reverse the other nodes
	            for i=1,smt.parentChildren[pid][2] do
	                preorder_tranverse(childIds[i])
	            end
	        end
	    end
	end  
	
	preorder_tranverse(smt.rootId)
	local pdist = probs:index(2,torch.LongTensor(leaf_node))
	print(pdist:sum(2))
	local _,pt = torch.max(pdist,2)
	pt = pt:squeeze()
	--local timer = torch.Timer()
	for sample = 1,nSample do
		pred[sample] = leaf_node[pt[sample]]
		--if sample ==1 then print('procces one sample cost '..timer:time().real..' second') end
	end	
	collectgarbage()
	return pred
end

return SmtInference