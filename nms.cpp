float iou(torch::Tensor const& box0, torch::Tensor const& box1)
{
	auto r0 = box0[3].item().toFloat() / 2;
	auto s0 = box0.slice(0, 0, 3) - r0;
	auto e0 = box0.slice(0, 0, 3) + r0;
	auto r1 = box1[3].item().toFloat() / 2;
	auto s1 = box1.slice(0, 0, 3) - r1;
	auto e1 = box1.slice(0, 0, 3) + r1;

	std::vector<float> overlap;
	for (int i = 0; i < 3; i++)
	{
		float tmin = min(e0[i].item().toFloat(), e1[i].item().toFloat());
		float tmax = max(s0[i].item().toFloat(), s1[i].item().toFloat());
		float t = max(0, (tmin - tmax));
		overlap.push_back(t);
	}
	float intersection = overlap[0] * overlap[1] * overlap[2];
	float un = box0[3].item().toFloat() * box0[3].item().toFloat() * box0[3].item().toFloat() + \
		box1[3].item().toFloat() * box1[3].item().toFloat() * box1[3].item().toFloat() - intersection;
	return intersection / un;
}

torch::Tensor like_nms(torch::Tensor const& data, float nms_th_thresh = 0.1)
{
	if (data.sizes()[0] <= 1) return data.clone();
	auto sort_index = torch::argsort(data.transpose(1, 0)[0], 0, true);
	auto pbb = torch::index_select(data, 0, sort_index);
	std::cout << "sort pbb: " << pbb << std::endl;

	std::vector<torch::Tensor> bboxes, outlist;
	bboxes.push_back(pbb[0]);
	outlist.push_back(pbb[0].unsqueeze(0));

	for (int i =1; i < pbb.sizes()[0]; i++)
	{
		auto bbox = pbb[i];
		bool flag = true;
		for (int j = 0; j < bboxes.size(); j++)
		{
			auto b0 = bbox.slice(0, 1, 5);
			auto b1 = bboxes[j].slice(0, 1, 5);
			if (iou(b0, b1) >= nms_th_thresh)
			{
				flag = false;
				break;
			}
		}
		if (flag)
		{
			bboxes.push_back(bbox);
			outlist.push_back(bbox.unsqueeze(0));
		}
	}
	
	
	auto output = torch::cat(outlist);
	return output;
}
