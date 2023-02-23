import * as dotenv from 'dotenv'
dotenv.config()

import express from 'express'
import fetch from 'node-fetch'
import bodyParser from 'body-parser'

const app = express()

app.use((req,res,next) => {
    next()
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
})


app.use(express.static('frontend/dist'))

app.get('/api/moderation', async (req, res) => {
    console.log('received get request', req.query)
    let r = await fetch(
        'https://api.openai.com/v1/moderations',
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.OPENAI_KEY}`
            },
            body: JSON.stringify({input: req.query.text})
        }
    )

    r = await r.json()

    res.json(r)
})


// config

const APP_PORT = process.env.APP_PORT ?? 8000

app.listen(APP_PORT, () => console.log(`listening on http://localhost:${APP_PORT}`))
