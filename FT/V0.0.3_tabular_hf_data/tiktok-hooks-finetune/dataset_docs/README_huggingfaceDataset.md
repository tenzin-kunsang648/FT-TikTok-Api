---
license: mit
task_categories:
- text-generation
tags:
- tiktok
- captions
- hooks
size_categories:
- 10K<n<100K
---

# Tiktok Caption and Hook Dataset
Grabbed the initial dataset from https://x.com/iamgdsa/status/1884294758484611336
Ran quick language classification atop it (probably is bad, but it gets the job done) , and created 3 new conversation columns:
1. `conversations` - based on given input variables, generate a full set of caption + hook
2. `conversations_caption` - based on given input variables including hook, generate a caption
3. `conversations_hook` - based on given input variables including caption, generate a hook

Enjoy!