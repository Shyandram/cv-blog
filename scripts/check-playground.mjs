/** Chapter coverage and mathematical-rendering smoke checks. Run after adding or renaming chapters. */
import { build } from 'esbuild';
import { readdirSync, readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
const bundled=await build({stdin:{contents:"export * from './src/lib/playground/coverage'; export * from './src/lib/playground/catalog';",resolveDir:process.cwd(),loader:'ts'},bundle:true,write:false,platform:'node',format:'esm'});
const {experiments,playgroundEntries,chapterExperiments,chaptersForExperiment,readingOnly}=await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`);
const ids=new Set(playgroundEntries.map(e=>e.id));assert.equal(ids.size,playgroundEntries.length,'Duplicate experiment ID');
const chapters=[];
for(const track of readdirSync('src/content/tutorials/en'))for(const file of readdirSync(`src/content/tutorials/en/${track}`)){
 if(!file.endsWith('.mdx')||file.startsWith('_'))continue;
 const body=readFileSync(`src/content/tutorials/en/${track}/${file}`,'utf8');if(/^draft:\s*true\s*$/m.test(body))continue;
 const slug=`${track}/${file.slice(0,-4)}`;chapters.push(slug);
 assert.ok(chapterExperiments(slug).length||readingOnly[slug],`Chapter has no experiment or deliberate reading-only reason: ${slug}`);
}
let states=0;
for(const e of experiments){
 assert.ok(chaptersForExperiment(e.id).some(slug=>chapters.includes(slug)),`No current chapter for ${e.id}`);
 assert.equal(new Set(e.controls.map(c=>c.key)).size,e.controls.length,`Duplicate control key in ${e.id}`);
 const defaults=Object.fromEntries(e.controls.map(c=>[c.key,c.value]));
 let variants=[defaults];
 for(const c of e.controls){
  if(c.kind==='range'){assert.ok(c.min<=Number(c.value)&&Number(c.value)<=c.max);variants=variants.flatMap(s=>[s,{...s,[c.key]:c.min},{...s,[c.key]:c.max}]);}
  else if(c.kind==='select')variants=variants.flatMap(s=>c.options.map(o=>({...s,[c.key]:o.value})));
 }
 for(const state of variants)for(const lang of ['en','zh-tw']){
  const result=e.compute(state,lang);states++;
  assert.ok(!/NaN|Infinity|undefined/.test(result.svg),`Invalid SVG in ${e.id}: ${JSON.stringify(state)}`);
  assert.ok(result.summary.trim()&&result.metrics.length,`Missing interpretation in ${e.id}`);
 }
}
console.log(JSON.stringify({chapters:chapters.length,withExperiments:chapters.filter(s=>chapterExperiments(s).length).length,readingOnly:chapters.filter(s=>readingOnly[s]).length,experiments:playgroundEntries.length,renderStates:states},null,2));
