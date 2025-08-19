import type { Plugin } from "@opencode-ai/plugin"
import { readFile, writeFile } from "fs/promises"
import { existsSync } from "fs"

const ALLOWED_TAGS = [
  "header", "main", "footer", "figure", "figcaption",
  "table", "thead", "tbody", "tfoot", "th", "tr", "td", "caption",
  "section", "nav", "aside", "p", "ul", "ol", "li", "h1",
  "h2", "h3", "h4", "h5", "h6", "img", "math", "code",
  "cite", "blockquote", "b", "i", "u", "s", "sup", "sub", "br"
]

// Inline style tags that don't affect leaf node status
const INLINE_STYLE_TAGS = new Set(["b", "i", "u", "s", "sup", "sub", "code", "cite", "br"])

/**
 * Parse a comma-separated range string into a list of integers.
 * 
 * @param rangeStr String like "0-3,5,7-9" or empty string
 * @returns Array of integers representing all tag ids in the ranges
 * @throws Error if the range string format is invalid
 */
function parseRangeString(rangeStr: string): number[] {
  if (!rangeStr.trim()) {
    return []
  }
  
  const ids: number[] = []
  const parts = rangeStr.split(",")
  
  for (const part of parts) {
    const trimmedPart = part.trim()
    if (trimmedPart.includes("-")) {
      // Handle range like "1-3"
      const rangeParts = trimmedPart.split("-", 2)
      if (rangeParts.length !== 2) {
        throw new Error(`Invalid range format: ${trimmedPart}`)
      }
      
      const startNum = parseInt(rangeParts[0].trim())
      const endNum = parseInt(rangeParts[1].trim())
      
      if (isNaN(startNum) || isNaN(endNum)) {
        throw new Error(`Non-numeric values in range: ${trimmedPart}`)
      }
      
      if (startNum > endNum) {
        throw new Error(`Invalid range: start (${startNum}) > end (${endNum})`)
      }
      
      for (let i = startNum; i <= endNum; i++) {
        ids.push(i)
      }
    } else {
      // Handle single number like "5"
      const num = parseInt(trimmedPart)
      if (isNaN(num)) {
        throw new Error(`Non-numeric value: ${trimmedPart}`)
      }
      ids.push(num)
    }
  }
  
  return ids.sort((a, b) => a - b)
}

/**
 * Simple HTML parser to extract elements and their attributes
 */
function parseHTML(html: string) {
  const elements: Array<{
    name: string
    attrs: Record<string, string>
    isLeafNode: boolean
  }> = []
  
  // Find all opening tags (both self-closing and regular)
  const openingTagRegex = /<(\w+)([^>]*?)(?:\s*\/>)/g
  let match
  
  while ((match = openingTagRegex.exec(html)) !== null) {
    const [fullMatch, tagName, attrsStr] = match
    
    // Parse attributes (including hyphenated ones like data-sources)
    const attrs: Record<string, string> = {}
    const attrRegex = /([\w-]+)=["']([^"']*)["']/g
    let attrMatch
    while ((attrMatch = attrRegex.exec(attrsStr)) !== null) {
      attrs[attrMatch[1]] = attrMatch[2]
    }
    
    // For simplicity, assume non-void elements could be leaf nodes
    // This is a simplified check - in real implementation we'd need better parsing
    const voidElements = ['img', 'br', 'hr', 'meta', 'link', 'area', 'base', 'col', 'embed', 'input', 'source', 'track', 'wbr']
    const isVoid = voidElements.includes(tagName.toLowerCase())
    
    elements.push({
      name: tagName,
      attrs,
      isLeafNode: !isVoid // Simplified assumption
    })
  }
  
  return elements
}

/**
 * Lightweight HTML tree parser sufficient for validation needs
 */
type ElementNode = {
  name: string
  attrs: Record<string, string>
  children: ElementNode[]
  parent?: ElementNode
}

function parseHTMLTree(html: string): { root: ElementNode, allNodes: ElementNode[], bodyNode?: ElementNode } {
  const root: ElementNode = { name: "__root__", attrs: {}, children: [] }
  const allNodes: ElementNode[] = []

  const tagRegex = /<\/?([a-zA-Z][\w:-]*)([^>]*?)\/?\s*>/g
  const attrRegex = /([\w-]+)=["']([^"']*)["']/g
  const voidElements = new Set(['img', 'br', 'hr', 'meta', 'link', 'area', 'base', 'col', 'embed', 'input', 'source', 'track', 'wbr'])

  const stack: ElementNode[] = [root]
  let matchTree: RegExpExecArray | null

  while ((matchTree = tagRegex.exec(html)) !== null) {
    const full = matchTree[0]
    const rawName = matchTree[1] || ""
    const tagName = rawName.toLowerCase()
    const attrsStr = matchTree[2] || ""
    const isClosing = full.startsWith("</")
    const isSelfClosing = full.endsWith("/>") || voidElements.has(tagName)

    if (isClosing) {
      // Pop until matching tag or root
      for (let i = stack.length - 1; i > 0; i--) {
        if (stack[i].name === tagName) {
          stack.length = i // pop matched element
          break
        }
      }
      continue
    }

    // Opening or self-closing tag
    const attrs: Record<string, string> = {}
    let am: RegExpExecArray | null
    while ((am = attrRegex.exec(attrsStr)) !== null) {
      attrs[am[1]] = am[2]
    }

    const node: ElementNode = { name: tagName, attrs, children: [], parent: stack[stack.length - 1] }
    stack[stack.length - 1].children.push(node)
    allNodes.push(node)

    if (!isSelfClosing) {
      stack.push(node)
    }
  }

  // Find body node if present
  let bodyNode: ElementNode | undefined
  for (const n of allNodes) {
    if (n.name === "body") { bodyNode = n; break }
  }

  return { root, allNodes, bodyNode }
}

/**
 * Collect all nodes in a subtree starting from the given root.
 */
function collectSubtree(root: ElementNode): ElementNode[] {
  const collected: ElementNode[] = []
  const stack: ElementNode[] = [root]
  while (stack.length > 0) {
    const current = stack.pop()
    if (!current) continue
    collected.push(current)
    for (let i = current.children.length - 1; i >= 0; i--) {
      stack.push(current.children[i])
    }
  }
  return collected
}

/**
 * Validate that output HTML properly covers all input IDs.
 */
async function validateHTMLStructure(inputFile: string, outputFile: string): Promise<{
  isValid: boolean
  message: string
}> {
  try {
    // Check if files exist
    if (!existsSync(inputFile)) {
      return { isValid: false, message: `Input file does not exist: ${inputFile}` }
    }
    
    if (!existsSync(outputFile)) {
      return { isValid: false, message: `Output file does not exist: ${outputFile}` }
    }
    
    // Read files
    const inputHTML = await readFile(inputFile, "utf-8")
    let outputHTML = await readFile(outputFile, "utf-8")
    
    // Replace <em> with <i> and <strong> with <b> in output
    const replacementsMade: string[] = []
    if (outputHTML.includes("<em>") || outputHTML.includes("</em>")) {
      outputHTML = outputHTML.replace(/<em>/g, "<i>").replace(/<\/em>/g, "</i>")
      replacementsMade.push("<em> → <i>")
    }
    if (outputHTML.includes("<strong>") || outputHTML.includes("</strong>")) {
      outputHTML = outputHTML.replace(/<strong>/g, "<b>").replace(/<\/strong>/g, "</b>")
      replacementsMade.push("<strong> → <b>")
    }
    
    // Write back the modified output if replacements were made
    if (replacementsMade.length > 0) {
      await writeFile(outputFile, outputHTML, "utf-8")
      console.warn(`📝 Auto-replaced tags: ${replacementsMade.join(", ")}`)
    }
    
    // Parse HTML
    const inputTree = parseHTMLTree(inputHTML)
    const outputTree = parseHTMLTree(outputHTML)
  
    // Extract IDs from input
    const idsInInput = new Set<number>()
    for (const element of inputTree.allNodes) {
      if (element.attrs.id) {
        const id = parseInt(element.attrs.id)
        if (isNaN(id)) {
          return { isValid: false, message: `Non-numeric ID found in input: ${element.attrs.id}` }
        }
        idsInInput.add(id)
      }
    }
    
    // Check for disallowed tags in output
    const disallowedTags = new Set<string>()
    // Only validate tags INSIDE <body> (exclude the <body> element itself).
    // If there's no <body>, fall back to validating the full document tree.
    const nodesToCheck: ElementNode[] = outputTree.bodyNode
      ? outputTree.bodyNode.children.flatMap(child => collectSubtree(child))
      : outputTree.allNodes
    for (const element of nodesToCheck) {
      if (!ALLOWED_TAGS.includes(element.name)) {
        disallowedTags.add(element.name)
      }
    }
    
    // Extract IDs from output data-sources attributes (leaf nodes only)
    const idsInOutput = new Set<number>()
    for (const element of outputTree.allNodes) {
      const dataSources = element.attrs["data-sources"]
      if (dataSources) {
        const hasNonInlineChild = element.children.some(child => !INLINE_STYLE_TAGS.has(child.name))
        if (!hasNonInlineChild) {
          try {
            const ids = parseRangeString(dataSources)
            ids.forEach(id => idsInOutput.add(id))
          } catch (error) {
            return { 
              isValid: false, 
              message: `Failed to parse data-sources '${dataSources}': ${error}` 
            }
          }
        }
      }
    }
    
    // Check coverage
    const missingIds = Array.from(idsInInput).filter(id => !idsInOutput.has(id))
    const extraIds = Array.from(idsInOutput).filter(id => !idsInInput.has(id))
    
    if (disallowedTags.size > 0 || missingIds.length > 0 || extraIds.length > 0) {
      const errorMsgs: string[] = []
      
      if (disallowedTags.size > 0) {
        errorMsgs.push(
          `You've used some HTML tags in the ${outputFile} file that are not allowed: ${Array.from(disallowedTags).sort()}.\n` +
          `The allowed tags are: ${ALLOWED_TAGS.join(", ")}.\n` +
          "Fix these tags and keep going! You're doing great!"
        )
      }
      
      if (missingIds.length > 0) {
        let missingMsg = `IDs in the ${inputFile} file not yet covered by leaf nodes in the ${outputFile} file: ${missingIds.sort()}. You'll need to add more leaf nodes (or make sure all existing leaf nodes have data-sources) that cover these ids.`
        if (missingIds.length < 30) {
          missingMsg += (
            "\n\nNote: if any input nodes are empty or contain garbage characters " +
            "you think shouldn't be in the final output, you may attach their ids " +
            "as data-sources to neighboring output nodes to pass the id validation " +
            "check. Please do this sparingly."
          )
        }
        errorMsgs.push(missingMsg)
      }
      
      if (extraIds.length > 0) {
        errorMsgs.push(
          "Output references IDs not present in input (invalid data-sources). " +
          `Extra IDs: ${extraIds.sort()}. ` +
          "Ensure data-sources only contain IDs that exist in the input and split ranges " +
          "to avoid spanning missing IDs."
        )
      }
      
      return { isValid: false, message: errorMsgs.join("; ") }
    }
    
    return { isValid: true, message: "All IDs properly covered" }
    
  } catch (error) {
    return { isValid: false, message: `Validation error: ${error}` }
  }
}

export const ValidateHtmlPlugin: Plugin = async ({ app, client, $ }) => {
  console.log("HTML Validation Plugin initialized!")

  return {
    "tool.execute.after": async ({ tool, sessionID, callID }, output) => {
      // Only validate when HTML files are written
      if (tool !== "write" && tool !== "edit") {
        console.log("Skipping HTML validation for non-write or edit tool", tool)
        return
      }
      
      const inputFile: string = "input.html"
      const outputFile: string = "output.html"
      
      const result = await validateHTMLStructure(inputFile, outputFile)
      
      let validationMessage = `🔍 **HTML Validation**: Checking restructuring of \`${inputFile}\` → \`${outputFile}\`\n\n`
      
      if (result.isValid) {
        // Check for placeholders and length
        const outputHTML = await readFile(outputFile, "utf-8")
        const inputHTML = await readFile(inputFile, "utf-8")
        
        const possiblePlaceholders: string[] = []
        const lines = outputHTML.split("\n")
        for (let i = 0; i < lines.length; i++) {
          if (/placeholder|<!--|\.\.\./.test(lines[i])) {
            possiblePlaceholders.push(`Line ${i + 1}: ${lines[i].trim()}`)
          }
        }
        
        validationMessage += `✅ **Validation Passed!** All IDs from input are present as data-sources in output and all tags are valid.`
        
        if (possiblePlaceholders.length > 0) {
          validationMessage += `\n\n⚠️ **Warning**: Possible placeholders detected. Please review these lines:\n${possiblePlaceholders.map(line => `- ${line}`).join("\n")}`
        }
        
        // If output file is <85% the length of the input file, add warning
        if (outputHTML.length < 0.85 * inputHTML.length) {
          validationMessage += `\n\n⚠️ **Length Warning**: Output is suspiciously short (${Math.round((outputHTML.length / inputHTML.length) * 100)}% of input length). Verify all content is properly represented.`
        }
        
        validationMessage += `\n\n**Next Steps**: Review the output HTML structure and semantic tags. If satisfied, mark the task complete.`
      } else {
        validationMessage += `📝 **Validation In Progress**: ${result.message}\n\nKeep going! You're making progress toward a valid HTML restructuring.`
      }
      
      // Modify the tool's displayed result instead of sending a new chat message
      console.log("Modifying tool's displayed result", validationMessage)
      output.title = "HTML Validation"
      output.output = `${output.output ?? ""}\n\n${validationMessage}`
      output.metadata = { ...(output.metadata || {}), htmlValidation: { inputFile, outputFile, isValid: result.isValid } }
    }
  }
}
