pull updates from the original Tencent repository into your fork anytime.

You'd add the original repo as a remote (usually called "upstream"), then fetch and merge changes from it. Your fork on GitHub acts as an intermediary - you can pull from upstream, merge locally, and push to your fork.

The typical workflow:
```
git remote add upstream https://github.com/Tencent-Hunyuan/HunyuanOCR.git
git fetch upstream
git merge upstream/main
```