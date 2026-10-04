import { createProcessor } from '@mdx-js/mdx';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import remarkCjkFriendly from 'remark-cjk-friendly';

const processors = {
  md: createProcessor({ format: 'md', remarkPlugins: [remarkMath, remarkGfm, remarkCjkFriendly] }),
  mdx: createProcessor({ format: 'mdx', remarkPlugins: [remarkMath, remarkGfm, remarkCjkFriendly] }),
};
interface TextNode { type: string; value?: string; children?: TextNode[] }

/** Count prose only, without evaluating MDX or counting source syntax as words. */
export function proseCount(body: string, format: 'md' | 'mdx') {
  const parts: string[] = [];
  function visit(node: TextNode) {
    if (node.type === 'text') parts.push(node.value ?? '');
    else node.children?.forEach(visit);
  }
  visit(processors[format].parse(body));
  const text = parts.join(' ');
  const characters = (text.match(/\p{Script=Han}/gu) ?? []).length;
  const words = (text.replace(/\p{Script=Han}/gu, ' ').match(/[\p{L}\p{N}]+(?:['’\-][\p{L}\p{N}]+)*/gu) ?? []).length;
  return { words, characters };
}
