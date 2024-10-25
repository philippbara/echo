from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/playground")
test = remote_chain.invoke({"input": "What is the person's name?"})

print(test)