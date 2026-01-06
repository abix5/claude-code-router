import { get_encoding } from "tiktoken";
import { sessionUsageCache, Usage } from "./cache";
import { readFile } from "fs/promises";
import { opendir, stat } from "fs/promises";
import { join } from "path";
import { CLAUDE_PROJECTS_DIR, HOME_DIR } from "@CCR/shared";
import { LRUCache } from "lru-cache";
import { ConfigService } from "../services/config";
import { TokenizerService } from "../services/tokenizer";

// Types from @anthropic-ai/sdk
interface Tool {
  name: string;
  description?: string;
  input_schema: object;
}

interface ContentBlockParam {
  type: string;
  [key: string]: any;
}

interface MessageParam {
  role: string;
  content: string | ContentBlockParam[];
}

interface MessageCreateParamsBase {
  messages?: MessageParam[];
  system?: string | any[];
  tools?: Tool[];
  [key: string]: any;
}

const enc = get_encoding("cl100k_base");

export const calculateTokenCount = (
  messages: MessageParam[],
  system: any,
  tools: Tool[]
) => {
  let tokenCount = 0;
  if (Array.isArray(messages)) {
    messages.forEach((message) => {
      if (typeof message.content === "string") {
        tokenCount += enc.encode(message.content).length;
      } else if (Array.isArray(message.content)) {
        message.content.forEach((contentPart: any) => {
          if (contentPart.type === "text") {
            tokenCount += enc.encode(contentPart.text).length;
          } else if (contentPart.type === "tool_use") {
            tokenCount += enc.encode(JSON.stringify(contentPart.input)).length;
          } else if (contentPart.type === "tool_result") {
            tokenCount += enc.encode(
              typeof contentPart.content === "string"
                ? contentPart.content
                : JSON.stringify(contentPart.content)
            ).length;
          }
        });
      }
    });
  }
  if (typeof system === "string") {
    tokenCount += enc.encode(system).length;
  } else if (Array.isArray(system)) {
    system.forEach((item: any) => {
      if (item.type !== "text") return;
      if (typeof item.text === "string") {
        tokenCount += enc.encode(item.text).length;
      } else if (Array.isArray(item.text)) {
        item.text.forEach((textPart: any) => {
          tokenCount += enc.encode(textPart || "").length;
        });
      }
    });
  }
  if (tools) {
    tools.forEach((tool: Tool) => {
      if (tool.description) {
        tokenCount += enc.encode(tool.name + tool.description).length;
      }
      if (tool.input_schema) {
        tokenCount += enc.encode(JSON.stringify(tool.input_schema)).length;
      }
    });
  }
  return tokenCount;
};

const getProjectSpecificRouter = async (
  req: any,
  configService: ConfigService
) => {
  // Check if there is project-specific configuration
  if (req.sessionId) {
    const project = await searchProjectBySession(req.sessionId);
    if (project) {
      const projectConfigPath = join(HOME_DIR, project, "config.json");
      const sessionConfigPath = join(
        HOME_DIR,
        project,
        `${req.sessionId}.json`
      );

      // First try to read sessionConfig file
      try {
        const sessionConfig = JSON.parse(await readFile(sessionConfigPath, "utf8"));
        if (sessionConfig && sessionConfig.Router) {
          return sessionConfig.Router;
        }
      } catch {}
      try {
        const projectConfig = JSON.parse(await readFile(projectConfigPath, "utf8"));
        if (projectConfig && projectConfig.Router) {
          return projectConfig.Router;
        }
      } catch {}
    }
  }
  return undefined; // Return undefined to use original configuration
};

/**
 * Resolve model alias to full "provider,model" format
 * @param modelName - Model name or alias (e.g., "fast" or "anthropic,claude-sonnet")
 * @param configService - Config service instance
 * @param logger - Logger instance
 * @returns Resolved "provider,model" string or null if not found
 */
const resolveModelAlias = (
  modelName: string,
  configService: ConfigService,
  logger: any
): string | null => {
  const providers = configService.get<any[]>("providers") || [];
  const modelAliases = configService.get<Record<string, string>>("modelAliases") || {};

  // 1. Check if already in "provider,model" format
  if (modelName.includes(",")) {
    const [provider, model] = modelName.split(",");
    const foundProvider = providers.find(p => p.name === provider);
    const foundModel = foundProvider?.models?.find(m => m === model);

    if (foundProvider && foundModel) {
      return modelName; // Valid provider,model
    }
    // Invalid provider,model - will fall back to Router scenarios
    logger.warn(
      `Invalid provider,model combination: '${modelName}', will use Router scenario`
    );
    return null;
  }

  // 2. Check user-defined aliases
  if (modelAliases[modelName]) {
    const aliasValue = modelAliases[modelName];

    // Validate that alias points to existing model
    if (aliasValue.includes(",")) {
      const [provider, model] = aliasValue.split(",");
      const foundProvider = providers.find(p => p.name === provider);
      const foundModel = foundProvider?.models?.find(m => m === model);

      if (foundProvider && foundModel) {
        logger.info(`Resolved alias '${modelName}' to '${aliasValue}'`);
        return aliasValue;
      } else {
        logger.warn(
          `Alias '${modelName}' points to non-existent model '${aliasValue}', will use Router scenario`
        );
        return null;
      }
    }
  }

  // 3. Search across all providers for this model name
  for (const provider of providers) {
    const foundModel = provider.models?.find((m: string) => m === modelName);
    if (foundModel) {
      const resolvedValue = `${provider.name},${foundModel}`;
      logger.info(`Resolved model '${modelName}' to '${resolvedValue}' from provider search`);
      return resolvedValue;
    }
  }

  // 4. Not found - return null to trigger Router scenario fallback
  return null;
};

/**
 * Helper function to create structured routing log
 */
const createRoutingLog = (
  originalModel: string,
  resolvedModel: string,
  scenarioType: RouterScenarioType,
  context: {
    tokenCount: number;
    sessionId?: string;
    thinking?: boolean;
    webSearch: boolean;
    longContext: boolean;
    projectSpecificRouter: boolean;
    aliasUsed: boolean;
  }
) => {
  const [provider, model] = resolvedModel.split(",");

  return {
    msg: "Model routing decision",
    routing: {
      originalModel,
      resolvedModel,
      provider,
      model,
      scenarioType,
      aliasUsed: context.aliasUsed,
    },
    context: {
      tokenCount: context.tokenCount,
      sessionId: context.sessionId || null,
      thinking: context.thinking || false,
      webSearch: context.webSearch,
      longContext: context.longContext,
      projectSpecificRouter: context.projectSpecificRouter,
    },
  };
};

const getUseModel = async (
  req: any,
  tokenCount: number,
  configService: ConfigService,
  lastUsage?: Usage | undefined
): Promise<{ model: string; scenarioType: RouterScenarioType }> => {
  const projectSpecificRouter = await getProjectSpecificRouter(req, configService);
  const providers = configService.get<any[]>("providers") || [];
  const Router = projectSpecificRouter || configService.get("Router");

  const originalModel = req.body.model;
  const hasWebSearch = Array.isArray(req.body.tools) &&
    req.body.tools.some((tool: any) => tool.type?.startsWith("web_search"));

  // NEW: Try to resolve alias first
  const resolved = resolveModelAlias(req.body.model, configService, req.log);

  if (resolved) {
    // Successfully resolved - validate and use
    const [provider, model] = resolved.split(",");
    const finalProvider = providers.find((p: any) => p.name === provider);
    const finalModel = finalProvider?.models?.find((m: any) => m === model);

    if (finalProvider && finalModel) {
      req.log.info(
        createRoutingLog(originalModel, resolved, 'default', {
          tokenCount,
          sessionId: req.sessionId,
          thinking: req.body.thinking,
          webSearch: hasWebSearch,
          longContext: false,
          projectSpecificRouter: !!projectSpecificRouter,
          aliasUsed: originalModel !== resolved && !originalModel.includes(","),
        })
      );
      return { model: resolved, scenarioType: 'default' };
    }
  }

  // If not resolved or invalid, continue with existing Router scenario logic...

  // if tokenCount is greater than the configured threshold, use the long context model
  const longContextThreshold = Router?.longContextThreshold || 60000;
  const lastUsageThreshold =
    lastUsage &&
    lastUsage.input_tokens > longContextThreshold &&
    tokenCount > 20000;
  const tokenCountThreshold = tokenCount > longContextThreshold;
  if ((lastUsageThreshold || tokenCountThreshold) && Router?.longContext) {
    req.log.info(
      createRoutingLog(originalModel, Router.longContext, 'longContext', {
        tokenCount,
        sessionId: req.sessionId,
        thinking: req.body.thinking,
        webSearch: hasWebSearch,
        longContext: true,
        projectSpecificRouter: !!projectSpecificRouter,
        aliasUsed: false,
      })
    );
    return { model: Router.longContext, scenarioType: 'longContext' };
  }
  // Check for subagent routing
  if (
    req.body?.system?.length > 1 &&
    req.body?.system[1]?.text?.startsWith("<CCR-SUBAGENT-MODEL>")
  ) {
    const match = req.body?.system[1].text.match(
      /<CCR-SUBAGENT-MODEL>(.*?)<\/CCR-SUBAGENT-MODEL>/s
    );
    if (match) {
      const subagentOriginalModel = match[1].trim();
      let subagentModel = subagentOriginalModel;

      // Try to resolve subagent alias
      const subagentResolved = resolveModelAlias(subagentModel, configService, req.log);
      const subagentAliasUsed = !!subagentResolved && subagentOriginalModel !== subagentResolved;

      if (subagentResolved) {
        subagentModel = subagentResolved;
      } else if (Router?.default) {
        // If alias resolution failed, fall back to Router.default
        req.log.warn(`Subagent model '${subagentModel}' could not be resolved, using Router.default`);
        subagentModel = Router.default;
      }

      req.body.system[1].text = req.body.system[1].text.replace(
        `<CCR-SUBAGENT-MODEL>${match[1]}</CCR-SUBAGENT-MODEL>`,
        ""
      );

      req.log.info(
        createRoutingLog(subagentOriginalModel, subagentModel, 'default', {
          tokenCount,
          sessionId: req.sessionId,
          thinking: req.body.thinking,
          webSearch: hasWebSearch,
          longContext: false,
          projectSpecificRouter: !!projectSpecificRouter,
          aliasUsed: subagentAliasUsed,
        })
      );
      return { model: subagentModel, scenarioType: 'default' };
    }
  }
  // Use the background model for any Claude Haiku variant
  const globalRouter = configService.get("Router");
  if (
    req.body.model?.includes("claude") &&
    req.body.model?.includes("haiku") &&
    globalRouter?.background
  ) {
    req.log.info(
      createRoutingLog(originalModel, globalRouter.background, 'background', {
        tokenCount,
        sessionId: req.sessionId,
        thinking: req.body.thinking,
        webSearch: hasWebSearch,
        longContext: false,
        projectSpecificRouter: !!projectSpecificRouter,
        aliasUsed: false,
      })
    );
    return { model: globalRouter.background, scenarioType: 'background' };
  }
  // The priority of websearch must be higher than thinking.
  if (
    Array.isArray(req.body.tools) &&
    req.body.tools.some((tool: any) => tool.type?.startsWith("web_search")) &&
    Router?.webSearch
  ) {
    req.log.info(
      createRoutingLog(originalModel, Router.webSearch, 'webSearch', {
        tokenCount,
        sessionId: req.sessionId,
        thinking: req.body.thinking,
        webSearch: true,
        longContext: false,
        projectSpecificRouter: !!projectSpecificRouter,
        aliasUsed: false,
      })
    );
    return { model: Router.webSearch, scenarioType: 'webSearch' };
  }
  // if exits thinking, use the think model
  if (req.body.thinking && Router?.think) {
    req.log.info(
      createRoutingLog(originalModel, Router.think, 'think', {
        tokenCount,
        sessionId: req.sessionId,
        thinking: true,
        webSearch: hasWebSearch,
        longContext: false,
        projectSpecificRouter: !!projectSpecificRouter,
        aliasUsed: false,
      })
    );
    return { model: Router.think, scenarioType: 'think' };
  }

  // Default fallback
  req.log.info(
    createRoutingLog(originalModel, Router?.default, 'default', {
      tokenCount,
      sessionId: req.sessionId,
      thinking: req.body.thinking,
      webSearch: hasWebSearch,
      longContext: false,
      projectSpecificRouter: !!projectSpecificRouter,
      aliasUsed: false,
    })
  );
  return { model: Router?.default, scenarioType: 'default' };
};

export interface RouterContext {
  configService: ConfigService;
  tokenizerService?: TokenizerService;
  event?: any;
}

export type RouterScenarioType = 'default' | 'background' | 'think' | 'longContext' | 'webSearch';

export interface RouterFallbackConfig {
  default?: string[];
  background?: string[];
  think?: string[];
  longContext?: string[];
  webSearch?: string[];
}

export const router = async (req: any, _res: any, context: RouterContext) => {
  const { configService, event } = context;
  // Parse sessionId from metadata.user_id
  if (req.body.metadata?.user_id) {
    const parts = req.body.metadata.user_id.split("_session_");
    if (parts.length > 1) {
      req.sessionId = parts[1];
    }
  }
  const lastMessageUsage = sessionUsageCache.get(req.sessionId);
  const { messages, system = [], tools }: MessageCreateParamsBase = req.body;
  const rewritePrompt = configService.get("REWRITE_SYSTEM_PROMPT");
  if (
    rewritePrompt &&
    system.length > 1 &&
    system[1]?.text?.includes("<env>")
  ) {
    const prompt = await readFile(rewritePrompt, "utf-8");
    system[1].text = `${prompt}<env>${system[1].text.split("<env>").pop()}`;
  }

  try {
    // Try to get tokenizer config for the current model
    const [providerName, modelName] = req.body.model.split(",");
    const tokenizerConfig = context.tokenizerService?.getTokenizerConfigForModel(
      providerName,
      modelName
    );

    // Use TokenizerService if available, otherwise fall back to legacy method
    let tokenCount: number;

    if (context.tokenizerService) {
      const result = await context.tokenizerService.countTokens(
        {
          messages: messages as MessageParam[],
          system,
          tools: tools as Tool[],
        },
        tokenizerConfig
      );
      tokenCount = result.tokenCount;
    } else {
      // Legacy fallback
      tokenCount = calculateTokenCount(
        messages as MessageParam[],
        system,
        tools as Tool[]
      );
    }

    let model;
    const customRouterPath = configService.get("CUSTOM_ROUTER_PATH");
    if (customRouterPath) {
      try {
        const customRouter = require(customRouterPath);
        req.tokenCount = tokenCount; // Pass token count to custom router
        model = await customRouter(req, configService.getAll(), {
          event,
        });
      } catch (e: any) {
        req.log.error(`failed to load custom router: ${e.message}`);
      }
    }
    if (!model) {
      const result = await getUseModel(req, tokenCount, configService, lastMessageUsage);
      model = result.model;
      req.scenarioType = result.scenarioType;
    } else {
      // Custom router doesn't provide scenario type, default to 'default'
      req.scenarioType = 'default';
    }
    req.body.model = model;
  } catch (error: any) {
    req.log.error(`Error in router middleware: ${error.message}`);
    const Router = configService.get("Router");
    req.body.model = Router?.default;
    req.scenarioType = 'default';
  }
  return;
};

// Memory cache for sessionId to project name mapping
// null value indicates previously searched but not found
// Uses LRU cache with max 1000 entries
const sessionProjectCache = new LRUCache<string, string>({
  max: 1000,
});

export const searchProjectBySession = async (
  sessionId: string
): Promise<string | null> => {
  // Check cache first
  if (sessionProjectCache.has(sessionId)) {
    const result = sessionProjectCache.get(sessionId);
    if (!result || result === '') {
      return null;
    }
    return result;
  }

  try {
    const dir = await opendir(CLAUDE_PROJECTS_DIR);
    const folderNames: string[] = [];

    // Collect all folder names
    for await (const dirent of dir) {
      if (dirent.isDirectory()) {
        folderNames.push(dirent.name);
      }
    }

    // Concurrently check each project folder for sessionId.jsonl file
    const checkPromises = folderNames.map(async (folderName) => {
      const sessionFilePath = join(
        CLAUDE_PROJECTS_DIR,
        folderName,
        `${sessionId}.jsonl`
      );
      try {
        const fileStat = await stat(sessionFilePath);
        return fileStat.isFile() ? folderName : null;
      } catch {
        // File does not exist, continue checking next
        return null;
      }
    });

    const results = await Promise.all(checkPromises);

    // Return the first existing project directory name
    for (const result of results) {
      if (result) {
        // Cache the found result
        sessionProjectCache.set(sessionId, result);
        return result;
      }
    }

    // Cache not found result (null value means previously searched but not found)
    sessionProjectCache.set(sessionId, '');
    return null; // No matching project found
  } catch (error) {
    console.error("Error searching for project by session:", error);
    // Cache null result on error to avoid repeated errors
    sessionProjectCache.set(sessionId, '');
    return null;
  }
};
