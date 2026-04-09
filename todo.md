下载https://github.com/karpathy/nanogpt，并上传到quietsmile的github进行管理。


参考repo: /newcpfs/user/yuchen/llm/cybertron_dots3.0_swa
增加一系列新的功能，包括:

0. 确定nanogpt可以进行bitwise determistic模式的训练
1. 把训练数据和validation数据都转换&保存成nanogpt可以读取的，确保训练的顺序跟cybertron完全一致. 各种细节也完全一致. 参考PAI训练任务(https://pai.console.aliyun.com/?regionId=ap-southeast-1&workspaceId=262162#/dlc/jobs/dlc1q9arre48b0kx/overview)
2. 把baseline的adam的训练任务的所有细节(模型结构，数据顺序，关键参数，lr策略等), 都跟cybertron的这个实验对齐。用deterministic的模式完成一次训练，看train loss是否非常接近(可以通过在pai上提交任务的方式进行)

注意：做事情要严谨，因为可以deterministic,所以不允许一点点对不齐。每一次有进展，都可以git commit，小步快跑。遇到复杂问题，先plan再执行。
 
