# System Design Note

## 1. Design Youtube
(credit to: http://blog.gainlo.co/index.php/2016/10/22/design-youtube-part/)

* Relational database (MySQL) or 
  - user model: can be one or two tables (email, username, registration data, profile information...)
  - video model: two tables, one for basic information and the others for more other details (title, description, size, count of likes, view counts, comments...)
  - author-video model: map user id to video id
  - user-like-video model: map user id to video id that the user likes

* Storage
  - store large static files seperately, most common way: CDN (content delivery network) to server content to end-users with high availability and high performance. CDN replicates content in multiple places, so the content may be closer to users. It takes care of scalability as well
  - popular videos store in CDN and less popular videos in local server
 
* Scalability
  - scale only when you need it. Start with a single server and later a single master with multiple read slaves. Then partition the database, such as partition by users' locations
  - for Youtube, can use two clusters, one more capable cluster for video and the other for general purpose

* Cache
  - server cache
  - front end cache
  
* Security
  - view hacking (send requests to hack the view count): check if one IP issues too many request or just restrict the number of view count per IP. Or check browser agent and users' past history
  
* Web Server
  - Youtube chooses Python which is faster to iterate and allows rapid flexible development and deployment
  - to scale the web server, simply have multiple replicas and build a load-balancer on top of them
  - server is for handling user requests and return reponse, other heavy logic should go to a seperate server such as recommnedation server which allows Python server to fetch data from
  
  
## 2. Cache System
* Concurrency
  - lock: affect the performance a lot
  - split the cache into multiple shards so that clients won't wait for each other if they are updating cache from different shards
  - commit logs, store the mutations into logs rather than update immediately. Then some background processes will execute all the logs asynchronously
  
* Distributed Cache
  - hash table: maps each resource to the corresponding machine. For example, when requesting resource A, we know machine M is responsible for cache A
  
## 3. Design TinyURL 
(credit to: https://www.youtube.com/watch?v=fMZMm_0ZhK4)

##### Generate Unique TinyUrl
* gerate random url
  - put into DB if absent, NoSQL is better in this situation because it scales up pretty well
  
* pick first 43 bits of MD5
  - same long URL would have same MD5 results, save spaces compared with random one. 
  - process: 0101010 convert to decimal, then convert to base 62
  
* Counter based:
  - single host: counter as maintainer, when get request from different worker host, it returns counter to it and then increase itself, so each worker host get unique counter. But single node, fail easily
  - all host: every host tries to maintain the counter 
  
##### Layers  
* Application layer
  - length need to be 6, both capital letter and lowercase letter and 0 - 9 (in total 62), if length is 7, then 62 ^ 7 = 3.5 trillion, if the server processes thoudands of url per second, it takes 110 years to consume. If millions of processing per second, then 40 days, it is be all used up

* Persistence layer


