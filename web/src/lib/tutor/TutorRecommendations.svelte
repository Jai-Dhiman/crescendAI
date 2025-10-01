<script lang="ts">
  export let feedback: import('$lib/api/tutor').TutorFeedback;
</script>

<div class="space-y-6">
  <div class="space-y-2">
    <h2 class="text-2xl font-semibold">Tutor Recommendations</h2>
    <p class="text-sm text-gray-600">Personalized, actionable suggestions with citations.</p>
  </div>

  {#if feedback.recommendations.length === 0}
    <div class="p-4 text-gray-600">No recommendations available.</div>
  {:else}
    <div class="space-y-4">
      {#each feedback.recommendations as rec, i}
        <div class="border rounded-lg p-4 bg-white space-y-2">
          <div class="flex items-start justify-between">
            <h3 class="text-lg font-medium">{rec.title}</h3>
            <span class="text-xs text-gray-500">~{rec.estimated_time_minutes} min</span>
          </div>
          <p class="text-gray-800">{rec.detail}</p>
          {#if rec.applies_to?.length}
            <div class="flex flex-wrap gap-2">
              {#each rec.applies_to as tag}
                <span class="px-2 py-0.5 text-xs rounded bg-gray-100 border">{tag}</span>
              {/each}
            </div>
          {/if}
          {#if rec.practice_plan?.length}
            <div>
              <div class="text-sm font-semibold mb-1">Practice plan</div>
              <ol class="list-decimal ml-5 space-y-1">
                {#each rec.practice_plan as step}
                  <li>{step}</li>
                {/each}
              </ol>
            </div>
          {/if}
          {#if rec.citations?.length}
            <div class="text-xs text-gray-600">
              Sources:
              {#each rec.citations as cid, j}
                <span>{cid}{j < rec.citations.length - 1 ? ', ' : ''}</span>
              {/each}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}

  <div class="space-y-2">
    <h3 class="text-lg font-semibold">Citations</h3>
    {#if feedback.citations.length === 0}
      <div class="p-2 text-sm text-gray-600">No citations provided.</div>
    {:else}
      <div class="space-y-2">
        {#each feedback.citations as c}
          <div class="border rounded p-3 bg-gray-50">
            <div class="text-sm font-medium">{c.id}: {c.title}</div>
            <div class="text-xs text-gray-600">{c.source}</div>
            {#if c.url}
              <a class="text-xs text-blue-600 underline" href={c.url} target="_blank" rel="noreferrer">{c.url}</a>
            {/if}
            {#if c.sections?.length}
              <div class="text-xs text-gray-700 mt-1">Sections: {c.sections.join(', ')}</div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>
