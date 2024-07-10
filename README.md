```
python3 -m venv .venv
```

```
source .venv/bin/activate
```

```
python3 -m pip install -r requirements.txt
```

```
uvicorn main:app --reload
```


```
docker build -t backend .
```

```
docker run -d -p 6002:6000 backend
```
