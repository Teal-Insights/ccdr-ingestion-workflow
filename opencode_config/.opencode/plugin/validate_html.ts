/// <reference types="bun" />
/// <reference lib="dom" />
import type { Plugin } from "@opencode-ai/plugin"
import { readFile, writeFile } from "fs/promises"
import { existsSync } from "fs"
import { resolve } from "path"

const ALLOWED_TAGS = [
  "header", "main", "footer", "figure", "figcaption",
  "table", "thead", "tbody", "tfoot", "th", "tr", "td", "caption",
  "section", "nav", "aside", "p", "ul", "ol", "li", "h1",
  "h2", "h3", "h4", "h5", "h6", "img", "math", "code",
  "cite", "blockquote", "b", "i", "u", "s", "sup", "sub", "br"
]

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
  const openingTagRegex = /<(\w+)([^>]*?)(?:\s*\/?>)/g
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
    const inputElements = parseHTML(inputHTML)
    const outputElements = parseHTML(outputHTML)
    
    // Extract IDs from input
    const idsInInput = new Set<number>()
    for (const element of inputElements) {
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
    for (const element of outputElements) {
      if (!ALLOWED_TAGS.includes(element.name)) {
        disallowedTags.add(element.name)
      }
    }
    
    // Extract IDs from output data-sources attributes (leaf nodes only)
    const idsInOutput = new Set<number>()
    for (const element of outputElements) {
      if (element.attrs["data-sources"] && element.isLeafNode) {
        try {
          const ids = parseRangeString(element.attrs["data-sources"])
          ids.forEach(id => idsInOutput.add(id))
        } catch (error) {
          return { 
            isValid: false, 
            message: `Failed to parse data-sources '${element.attrs["data-sources"]}': ${error}` 
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
    "tool.execute.after": async (input, output) => {
      // Only validate when HTML files are written
      if (input.tool === "write" && (output as any).args?.file_path?.endsWith(".html")) {
        const outputFile = resolve((output as any).args.file_path)
        
        // Look for a corresponding input file
        // Try common patterns: input.html, original.html, or same directory with "input" prefix
        const outputDir = outputFile.substring(0, outputFile.lastIndexOf("/"))
        const possibleInputFiles = [
          resolve(outputDir, "input.html"),
          resolve(outputDir, "original.html"),
          outputFile.replace(/\/([^/]+)\.html$/, "/input.html"),
          outputFile.replace(/\/output\.html$/, "/input.html")
        ]
        
        let inputFile: string | null = null
        for (const candidate of possibleInputFiles) {
          if (existsSync(candidate)) {
            inputFile = candidate
            break
          }
        }
        
        if (!inputFile) {
          // Don't error if no input file found - this might not be a restructuring task
          return
        }
        
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
        
        // Use system-reminder tag to ensure the LLM pays attention to validation feedback
        const systemReminder = `<system-reminder>${validationMessage}</system-reminder>`
        
        // Modify the tool output to include validation results
        output.output = `${output.output}\n\n${systemReminder}`
        output.metadata = {
          ...output.metadata,
          htmlValidation: {
            isValid: result.isValid,
            message: result.message,
            inputFile,
            outputFile
          }
        }
      }
    }
  }
}