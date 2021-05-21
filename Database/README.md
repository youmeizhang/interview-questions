## Database

### Sharding
It is a database architecture pattern related to horizontal partitioning. It seprates the table's row into multiple different tables known as partitions. Each partition has the same schema and columns, but also entirely different rows. 

#### Vertical Partition vs Horizontal Partition
* Vertical Partition
  * entire columns are seperated out and put into new, distinct tables
* Horizontal Partition

#### Pros
* Horizontal scaling (scaling out)
  * add more machines to an existing stack in order to spread out the load and allow more traffic and faster processing
* Speed up query response time
* Make application more reliable by mitigating the impact of outages

#### Cons
* More complex to implement
* Shards would become unbalanced 
* Once a db is sharded, it is difficult to return to its unsharded architecture
* Sharding is not natively supported by every database engine

#### Sharding Architecture
* Key based sharding
  * Use a value taken from newly written data and plug it into a hash function to determine which shard the data should go to
  * Simple case: shard keys are similar to primary keys. But shard key should be static
  * But for adding / removing servers to a database, need to remap to new correct hash value and then migrate to the appropriate server
* Range based sharding
  * Sharding data based on ranges of given value
  * It can still cause unevenly distributed data and thus leading to database hotsopts
* Directory based sharding
  * Maintain a lookup table that use shard key to keep track of which shard holds which data
  * Flexible. Range based sharding limit you to specifying ranges of values while key based ones limit you to use a fixed hash function. Directory based sharding allows you to use whatever system or algorithm you want to assign data entries to shards
  * However, need to connect to the lookup table before each query or write which can have a determental impact on an application's performance
  * Lookup table becomes a single point of failure

#### Options for optimizing your database
* Setting up a remote database
* Implementing caching
* Creating one or more read replicas
* Upgrading to a larger server

#### redis如何解决热点问题
1. 对于“get”类型的热点key，通常可以为redis添加slave，通过slave承担读压力来缓解
2. 服务端本地缓存，服务端先请求本地缓存，缓解redis压力
3. 多级缓存方案，通过多级请求，层层过滤解决热点key问题
4. proxy方案，有些方案会探测分片热点key，缓存在proxy上缓解redis压力
5. 同解决big方案类似，将一个key通过hash分解为多个key，value值一样，将这些key分散到集群多个分片中，需要访问时先根据hash算出对应的key，然后访问的具体的分片
