<script>

import { fly } from 'svelte/transition'
import { onMount } from 'svelte'
import Results from './lib/Results.svelte'

let evaluation_state = 'none'
let unsafe_text = ''

const API_ENDPOINT = '/api/moderation'

onMount(() => {
    unsafe_text = localStorage.getItem('unsafe_text') ?? ''
})

const save_text = () => localStorage.setItem('unsafe_text', unsafe_text)

const submit = () => {
    save_text()
    evaluation_state = 'waiting'
}

const reset = () => {
    evaluation_state = 'none'
}

const evaluate = async () => {

    let r = await fetch(`${API_ENDPOINT}?text=${encodeURIComponent(unsafe_text)}`)
    r = await r.json()

    console.log(r)

    evaluation_state = 'complete'

    return r
}

</script>

<main>
    <h1>AIST Moderation</h1>
    
    <p>
        This research tool is designed to evaluate user-provided text for
        potentially harmful content. This includes identifying explicit
        language, hate speech, and other undesirable model behaviors. It is
        important to note that this tool is strictly for research purposes and
        is not intended for commercial or legal use.
    </p>
    
    <div class="group">
        <label for="input_area">Input text</label>
        <textarea
            id="input_area"
            on:keyup={save_text}
            bind:value={unsafe_text}
            name="input_area"></textarea>
    </div>

    <div class="button-row">
        {#if evaluation_state == 'complete'}
            <button
                class="button-reset"
                type="button"
                transition:fly={{ x: -100 }}
                on:click={reset}>
                    <span>Reset</span>
                    <i class="fa-solid fa-arrows-rotate"></i>
            </button>
        {/if}
        
        <button class="button-submit" class:disabled={evaluation_state != 'none'} type="button" on:click={submit}>
            <span>Evaluate</span>
            <i class="fa-solid fa-arrow-right"></i>
        </button>
    </div>

    {#if evaluation_state != 'none'}
        {#await evaluate()}
            <div 
                transition:fly={{ y: 200 }}
                class="loading">
                <i class="fa-solid fa-spinner"></i>
                <span>Waiting for a response...</span>
            </div>
        {:then result} 
            <div 
                transition:fly={{ y: 200 }}>
                <Results
                    response={result}/>
            </div>
        {/await}
    {/if}

</main>

