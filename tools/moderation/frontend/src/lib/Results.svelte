<script>

export let response //= {"id":"modr-6mr8Gpo7Cj8SlxIPuS93AXmgnV0MO","model":"text-moderation-004","results":[{"flagged":false,"categories":{"sexual":false,"hate":false,"violence":false,"self-harm":false,"sexual/minors":false,"hate/threatening":false,"violence/graphic":false},"category_scores":{"sexual":0.000005492542186402716,"hate":0.000002904485427279724,"violence":0.000001915554321385571,"self-harm":9.2021507080986e-10,"sexual/minors":1.1055483639665908e-8,"hate/threatening":9.075322437990252e-11,"violence/graphic":1.250628685056654e-7}}]}

const convert_to_triples = res => {
    let out = [
        ['flagged', `${res.flagged}`, 'result']
    ]

    for (let key in res.categories) {
        out.push([
            `categories/${key}`, `${res.categories[key]}`, 'result'
        ])
    }

    for (let key in res.categories) {
        out.push([
            `category_scores/${key}`, `${res.category_scores[key]}`, 'result'
        ])
    }
    
    return out
}

</script>

<h2>Model results</h2>

{#each response.results as result}
    <table class="table">
        <thead>
            <tr>
                <th>Key</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
            {#each convert_to_triples(result) as row}
                <tr>
                    <td>{row[0]}</td>
                    <td class={row[2]}>{row[1]}</td>
                </tr>
            {/each}
        </tbody>
    </table>
    
{/each}

<details>
    <summary>Raw output</summary>
    <div class="pre-wrap">
        <pre>{JSON.stringify(response, null, 4)}</pre>
    </div>
</details>

<style>
.pre-wrap {
    font-size: 10pt;
    font-family: 'Inconsolata', monospace;
}

summary {
    color: var(--dark-gray);
    cursor: pointer;
    user-select: none;
}

.table {
    width: 100%;
}

td:last-child {
    font-family: var(--font-monospace);
}

th {
    text-align: left;
}

details, .table {
    margin: 1rem 0;
}
</style>
