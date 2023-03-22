# MangoMIL
 Multi stain histolopathology graph based deep Multiple Instance Leanrning:
 
 <ol>
  <li>An embedding backbone reduces each patch to a 1024 feature vector. </li>
  <li>All embeddings from a given patient are aggregated into a larger feature vector</li>
  <li>The feature vector is used to create a k-nearest neighbour graph (k=5, distance=minkoswki)</li>
  <li>Successive Graph Self Attention layers (D1) and Self Attention Graph Pooling layers (D2) are applied to the sparse graph</li>
  <li>Finally the graph is classified</li>
</ol>

![alt text](https://github.com/AmayaGS/MangoMIL/blob/main/graph_model.png?raw=true)
