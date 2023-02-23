# moderation tool

This is a small wrapper for OpenAI's moderation endpoint, including a frontend
for testing.

## how to use

First, get an API key from [OpenAI](//https://openai.com/api/) and stick it in 

The easiest way to get started is using Docker. First, clone the repository:

```
$ git clone https://github.com/pkage/ai-storytelling-backstage
$ cd ai-storytelling-backstage/tools/moderation
```

Then, populate the `.env` file with your API key:

```
$ cp template.env .env
$ $EDITOR .env
```

Then you can just run:

```
$ docker-compose up
```

Navigate to [localhost:8000](http://localhost:8000) for a demo!


