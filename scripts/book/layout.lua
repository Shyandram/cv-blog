-- Keep equations native LaTeX; shrink only displays wider than the text measure.
function Math(el)
  el.text = el.text:gsub('\\setminus', '\\smallsetminus')
  if el.mathtype == 'DisplayMath' then
    local s = el.text:gsub('\\lt([^%a])', '<%1'):gsub('\\gt([^%a])', '>%1')
    return pandoc.RawInline('latex', '\\bookdisplay{' .. s .. '}')
  elseif #el.text > 100 then
    return pandoc.RawInline('latex', '\\bookinline{' .. el.text .. '}')
  end
  el.text = el.text:gsub(',', ',\\allowbreak ')
  return el
end
function CodeBlock(el)
  return pandoc.RawBlock('latex', '\\begin{Verbatim}\n' .. el.text .. '\n\\end{Verbatim}')
end
local function escape(text)
  return text:gsub('[\\{}%%$&#_~^]', function(c)
    local codes = {['\\']='\\textbackslash{}', ['{']='\\{', ['}']='\\}', ['%']='\\%', ['$']='\\$', ['&']='\\&', ['#']='\\#', ['_']='\\_', ['~']='\\textasciitilde{}', ['^']='\\textasciicircum{}'}
    return codes[c]
  end)
end
function Code(el)
  local chars = {}
  for _, code in utf8.codes(el.text) do chars[#chars + 1] = escape(utf8.char(code)) end
  return pandoc.RawInline('latex', '\\texttt{' .. table.concat(chars, '\\allowbreak{}') .. '}')
end
function Str(el)
  if el.text:find('/') or el.text:find('_') or el.text:find('doi:') or el.text:find('arXiv:') then
    local s = escape(el.text):gsub('\\_', '\\_\\allowbreak{}'):gsub('/', '/\\allowbreak{}'):gsub('%.', '.\\allowbreak{}'):gsub(':', ':\\allowbreak{}')
    return pandoc.RawInline('latex', s)
  end
end
function Image(el)
  return pandoc.RawInline('latex', '\\bookimage{' .. el.src .. '}')
end

-- Markdown source widths include link destinations; use balanced print columns.
function Table(el)
  for i, spec in ipairs(el.colspecs) do
    el.colspecs[i] = {spec[1], 1 / #el.colspecs}
  end
  return el
end
