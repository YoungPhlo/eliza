import { Tweet } from "agent-twitter-client";
import {
    composeContext,
    generateText,
    embeddingZeroVector,
    IAgentRuntime,
    ModelClass,
    ModelProviderName,
    stringToUuid,
    elizaLogger,
} from "@ai16z/eliza";
import { ClientBase } from "./base.ts";
import * as fs from 'fs/promises';

const twitterPostTemplate = `{{timeline}}

# Knowledge
{{knowledge}}

About {{agentName}} (@{{twitterUserName}}):
{{bio}}
{{lore}}
{{postDirections}}

{{providers}}

{{recentPosts}}

{{characterPostExamples}}

# Task: Generate a post in the voice and style of {{agentName}}, aka @{{twitterUserName}}
Write a single sentence post that is {{adjective}} about {{topic}} (without mentioning {{topic}} directly), from the perspective of {{agentName}}. Try to write something totally different than previous posts. Do not add commentary or acknowledge this request, just write the post.
Your response should not contain any questions. Brief, concise statements only. No emojis. Use \\n\\n (double spaces) between statements.`;

const MAX_TWEET_LENGTH = 280;

/**
 * Truncate text to fit within the Twitter character limit, ensuring it ends at a complete sentence.
 */
function truncateToCompleteSentence(text: string): string {
    if (text.length <= MAX_TWEET_LENGTH) {
        return text;
    }

    // Attempt to truncate at the last period within the limit
    const truncatedAtPeriod = text.slice(
        0,
        text.lastIndexOf(".", MAX_TWEET_LENGTH) + 1
    );
    if (truncatedAtPeriod.trim().length > 0) {
        return truncatedAtPeriod.trim();
    }

    // If no period is found, truncate to the nearest whitespace
    const truncatedAtSpace = text.slice(
        0,
        text.lastIndexOf(" ", MAX_TWEET_LENGTH)
    );
    if (truncatedAtSpace.trim().length > 0) {
        return truncatedAtSpace.trim() + "...";
    }

    // Fallback: Hard truncate and add ellipsis
    return text.slice(0, MAX_TWEET_LENGTH - 3).trim() + "...";
}

// Format duration in milliseconds to human readable string
function formatDuration(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    const parts: string[] = [];
    
    if (hours > 0) {
        parts.push(`${hours}h`);
    }
    if (minutes % 60 > 0) {
        parts.push(`${minutes % 60}m`);
    }
    if (seconds % 60 > 0 || parts.length === 0) {
        parts.push(`${seconds % 60}s`);
    }
    
    return parts.join(' ');
}

// Performance and monitoring types
interface StageMetrics {
    startTime: number;
    endTime?: number;
    duration?: number;
    success: boolean;
    error?: string;
    attempts: number;
    retryCount: number;
}

interface TweetGenerationMetrics {
    stages: {
        initialization?: StageMetrics;
        textGeneration?: StageMetrics;
        imageGeneration?: StageMetrics;
        posting?: StageMetrics;
    };
    totalDuration?: number;
    overallSuccess: boolean;
    scheduledTime?: number;
    actualPostTime?: number;
    delay?: number;
}

interface TweetGenerationStatus {
    isGenerating: boolean;
    startTime?: number;
    currentStage?: 'initialization' | 'text' | 'image' | 'posting';
    attempts: number;
    lastError?: string;
    metrics: TweetGenerationMetrics;
    retryCount: number;
    lastSuccessfulPost?: number;
    consecutiveFailures: number;
}

export class TwitterPostClient {
    client: ClientBase;
    runtime: IAgentRuntime;

    private tweetGenerationStatus: TweetGenerationStatus = {
        isGenerating: false,
        attempts: 0,
        metrics: {
            stages: {},
            overallSuccess: false
        },
        retryCount: 0,
        consecutiveFailures: 0
    };

    private readonly MAX_RETRY_ATTEMPTS = 3;
    private readonly RETRY_DELAY_MS = 30000; // 30 seconds

    private startStageMetrics(stage: keyof TweetGenerationMetrics['stages']) {
        this.tweetGenerationStatus.metrics.stages[stage] = {
            startTime: Date.now(),
            success: false,
            attempts: 1,
            retryCount: 0
        };
        this.logPerformanceMetrics(`Starting ${stage}`);
    }

    private endStageMetrics(stage: keyof TweetGenerationMetrics['stages'], success: boolean, error?: Error) {
        const stageMetrics = this.tweetGenerationStatus.metrics.stages[stage];
        if (stageMetrics) {
            stageMetrics.endTime = Date.now();
            stageMetrics.duration = stageMetrics.endTime - stageMetrics.startTime;
            stageMetrics.success = success;
            if (error) {
                stageMetrics.error = error.message;
            }
            this.logPerformanceMetrics(`Completed ${stage}`, success, error);
        }
    }

    private logPerformanceMetrics(action: string, success?: boolean, error?: Error) {
        const metrics = this.tweetGenerationStatus.metrics;
        const currentStage = this.tweetGenerationStatus.currentStage;
        const stageMetrics = currentStage ? metrics.stages[currentStage] : undefined;

        const logData = {
            action,
            timestamp: new Date().toISOString(),
            currentStage,
            totalElapsedTime: this.tweetGenerationStatus.startTime 
                ? Math.floor((Date.now() - this.tweetGenerationStatus.startTime) / 1000)
                : 0,
            stageElapsedTime: stageMetrics?.startTime 
                ? Math.floor((Date.now() - stageMetrics.startTime) / 1000)
                : 0,
            attempts: this.tweetGenerationStatus.attempts,
            retryCount: this.tweetGenerationStatus.retryCount,
            consecutiveFailures: this.tweetGenerationStatus.consecutiveFailures,
            scheduledDelay: metrics.delay,
            stages: Object.entries(metrics.stages).map(([name, stats]) => ({
                name,
                duration: stats.duration ? formatDuration(stats.duration) : undefined,
                success: stats.success,
                attempts: stats.attempts
            }))
        };

        if (error) {
            elizaLogger.error('Tweet generation performance metrics:', {
                ...logData,
                error: {
                    message: error.message,
                    stack: error.stack,
                    name: error.name
                }
            });
        } else {
            elizaLogger.log('Tweet generation performance metrics:', logData);
        }
    }

    private async handleStageRetry(
        stage: keyof TweetGenerationMetrics['stages'],
        error: Error,
        retryAction: () => Promise<void>
    ) {
        const stageMetrics = this.tweetGenerationStatus.metrics.stages[stage];
        if (stageMetrics) {
            stageMetrics.retryCount++;
            stageMetrics.attempts++;
            
            if (stageMetrics.retryCount <= this.MAX_RETRY_ATTEMPTS) {
                elizaLogger.warn(`Retrying ${stage} after error:`, {
                    error: error.message,
                    attempt: stageMetrics.attempts,
                    retryCount: stageMetrics.retryCount
                });
                
                await new Promise(resolve => setTimeout(resolve, this.RETRY_DELAY_MS));
                await retryAction();
            } else {
                throw new Error(`Max retry attempts (${this.MAX_RETRY_ATTEMPTS}) exceeded for ${stage}`);
            }
        }
    }

    async start(postImmediately: boolean = false) {
        if (!this.client.profile) {
            await this.client.init();
        }

        const generateNewTweetLoop = async () => {
            const lastPost = await this.runtime.cacheManager.get<{
                timestamp: number;
            }>(
                "twitter/" +
                    this.runtime.getSetting("TWITTER_USERNAME") +
                    "/lastPost"
            );

            const now = Date.now();
            const lastPostTimestamp = lastPost?.timestamp ?? now;
            const minMinutes = parseInt(this.runtime.getSetting("POST_INTERVAL_MIN")) || 90;
            const maxMinutes = parseInt(this.runtime.getSetting("POST_INTERVAL_MAX")) || 180;
            const randomMinutes = Math.floor(Math.random() * (maxMinutes - minMinutes + 1)) + minMinutes;
            const delay = randomMinutes * 60 * 1000;

            // Get the cached next tweet time if it exists
            const nextTweetTimeCache = await this.runtime.cacheManager.get<{
                timestamp: number;
                scheduledAt: number;
            }>("twitter/" + this.runtime.getSetting("TWITTER_USERNAME") + "/nextTweetTime");

            // Debug log the timing information
            elizaLogger.debug("Tweet timing debug:", {
                now: new Date(now).toISOString(),
                lastPostTimestamp: new Date(lastPostTimestamp).toISOString(),
                nextTweetTimeCache: nextTweetTimeCache ? {
                    timestamp: new Date(nextTweetTimeCache.timestamp).toISOString(),
                    scheduledAt: new Date(nextTweetTimeCache.scheduledAt).toISOString()
                } : null,
                randomMinutes,
                delay: formatDuration(delay)
            });

            // Use cached next tweet time or calculate from last post
            const nextTweetTime = nextTweetTimeCache?.timestamp || (lastPostTimestamp + delay);

            if (now >= nextTweetTime) {
                const executionStart = Date.now();
                
                // Calculate delay relative to cached schedule or last post
                let delaySeconds = 0;
                if (nextTweetTimeCache) {
                    // If we have a cached schedule, calculate delay from that
                    delaySeconds = Math.max(0, Math.floor((now - nextTweetTimeCache.timestamp) / 1000));
                } else {
                    // Otherwise calculate from last post plus minimum interval
                    const minimumNextTime = lastPostTimestamp + (minMinutes * 60 * 1000);
                    delaySeconds = Math.max(0, Math.floor((now - minimumNextTime) / 1000));
                }

                elizaLogger.log(`Tweet time reached (${delaySeconds} seconds past scheduled time), generating new tweet...`);
                
                try {
                    await this.generateNewTweet();
                    const executionEnd = Date.now();
                    const executionDuration = executionEnd - executionStart;
                    
                    // Log performance metrics
                    elizaLogger.log("Tweet generation performance:", {
                        totalTime: formatDuration(executionDuration),
                        stages: {
                            initialization: formatDuration(this.tweetGenerationStatus.metrics.stages.initialization?.duration || 0),
                            textGeneration: formatDuration(this.tweetGenerationStatus.metrics.stages.textGeneration?.duration || 0),
                            imageGeneration: this.tweetGenerationStatus.metrics.stages.imageGeneration ? 
                                formatDuration(this.tweetGenerationStatus.metrics.stages.imageGeneration.duration || 0) : 'skipped',
                            posting: formatDuration(this.tweetGenerationStatus.metrics.stages.posting?.duration || 0)
                        },
                        attempts: this.tweetGenerationStatus.attempts,
                        retryCount: this.tweetGenerationStatus.retryCount,
                        scheduledDelay: delaySeconds > 0 ? `${formatDuration(delaySeconds * 1000)} past scheduled time` : 'on time'
                    });

                    // Schedule next tweet from NOW
                    const newNextTweetTime = now + delay;
                    
                    // Debug log the new schedule
                    elizaLogger.debug("New tweet schedule:", {
                        executionDuration: formatDuration(executionDuration),
                        newNextTweetTime: new Date(newNextTweetTime).toISOString(),
                        delay: formatDuration(delay),
                        delayFromSchedule: formatDuration(delaySeconds * 1000)
                    });

                    await this.runtime.cacheManager.set(
                        "twitter/" + this.runtime.getSetting("TWITTER_USERNAME") + "/nextTweetTime",
                        { 
                            timestamp: newNextTweetTime,
                            scheduledAt: now,
                            intervalMinutes: randomMinutes,
                            lastExecutionDuration: executionDuration,
                            lastDelay: delaySeconds
                        }
                    );
                    
                    const nextTime = new Date(newNextTweetTime);
                    elizaLogger.log(`Next tweet scheduled for ${nextTime.toLocaleTimeString()} (in ${randomMinutes} minutes)`);
                    
                    // Set up next check in exactly the calculated delay
                    setTimeout(() => generateNewTweetLoop(), delay);
                } catch (error) {
                    // If tweet generation fails, retry sooner
                    elizaLogger.error("Tweet generation failed, retrying in 5 minutes", error);
                    setTimeout(() => generateNewTweetLoop(), 5 * 60 * 1000);
                }
            } else {
                // Calculate exact time until next tweet
                const timeUntilTweet = nextTweetTime - now;
                const minutesUntilTweet = Math.ceil(timeUntilTweet / (60 * 1000));
                
                await this.runtime.cacheManager.set(
                    "twitter/" + this.runtime.getSetting("TWITTER_USERNAME") + "/nextTweetTime",
                    { 
                        timestamp: nextTweetTime,
                        scheduledAt: now,
                        intervalMinutes: randomMinutes
                    }
                );
                
                const nextTime = new Date(nextTweetTime);
                elizaLogger.log(`Next tweet scheduled for ${nextTime.toLocaleTimeString()} (in ${minutesUntilTweet} minutes)`);
                
                // Set up next check at exactly the right time
                setTimeout(() => generateNewTweetLoop(), timeUntilTweet);
            }
        };

        if (postImmediately) {
            this.generateNewTweet();
        }

        generateNewTweetLoop();
    }

    constructor(client: ClientBase, runtime: IAgentRuntime) {
        this.client = client;
        this.runtime = runtime;
    }

    /**
     * Convert a data URL or file path to a Buffer
     * @param input The data URL string or file path
     * @returns Buffer containing the image data
     */
    private async imageToBuffer(input: string): Promise<Buffer> {
        // Check if it's a data URL
        if (input.startsWith('data:')) {
            const matches = input.match(/^data:([A-Za-z-+\/]+);base64,(.+)$/);
            if (!matches || matches.length !== 3) {
                throw new Error('Invalid data URL');
            }
            return Buffer.from(matches[2], 'base64');
        }
        
        // Otherwise treat it as a file path
        return fs.readFile(input);
    }

    private async generateNewTweet() {
        // Initialize metrics for new tweet generation
        this.tweetGenerationStatus = {
            isGenerating: true,
            startTime: Date.now(),
            attempts: this.tweetGenerationStatus.attempts + 1,
            currentStage: 'initialization',
            metrics: {
                stages: {},
                overallSuccess: false,
                scheduledTime: Date.now()
            },
            retryCount: 0,
            consecutiveFailures: this.tweetGenerationStatus.consecutiveFailures,
        };

        try {
            // Initialization stage
            this.startStageMetrics('initialization');
            elizaLogger.log(`Starting tweet generation attempt ${this.tweetGenerationStatus.attempts}`);
            this.endStageMetrics('initialization', true);

            // Text generation stage
            this.tweetGenerationStatus.currentStage = 'text';
            this.startStageMetrics('textGeneration');
            const content = await this.generateTweetText();
            if (!content) {
                throw new Error('Failed to generate tweet text - no content returned');
            }
            this.endStageMetrics('textGeneration', true);

            // Image generation decision
            const rawChance = this.runtime.getSetting("IMAGE_GEN_CHANCE") || "30";
            const imageGenChancePercent = parseFloat(rawChance.replace(/[^0-9.]/g, '')) || 30;
            elizaLogger.log(`Image generation chance set to ${imageGenChancePercent}%`);
            
            const shouldGenerateImage = Math.random() < (Math.max(0, Math.min(100, imageGenChancePercent)) / 100);
            elizaLogger.log(`Will ${shouldGenerateImage ? '' : 'not '}generate image for this tweet`);

            let imageBuffer: Buffer | undefined;
            if (shouldGenerateImage) {
                this.tweetGenerationStatus.currentStage = 'image';
                this.startStageMetrics('imageGeneration');
                try {
                    const imagePrompt = `Generate an image that represents this tweet: ${content}`;
                    
                    // Generate image using the plugin
                    const imageAction = this.runtime.plugins.find(p => p.name === "imageGeneration")?.actions?.[0];
                    if (!imageAction?.handler) {
                        throw new Error('Image generation plugin not found or handler not available');
                    }

                    // Temporarily set modelProvider to HEURIST for image generation
                    const originalProvider = this.runtime.character.modelProvider;
                    const originalToken = this.runtime.token;
                    
                    // Switch to HEURIST provider and set HEURIST API key
                    this.runtime.character.modelProvider = ModelProviderName.HEURIST;
                    this.runtime.token = this.runtime.getSetting("HEURIST_API_KEY");

                    try {
                        const state = await this.runtime.composeState({
                            userId: this.runtime.agentId,
                            roomId: stringToUuid("twitter_image_generation"),
                            agentId: this.runtime.agentId,
                            content: {
                                text: imagePrompt,
                                action: "GENERATE_IMAGE",
                                payload: {
                                    prompt: imagePrompt,
                                    model: this.runtime.getSetting("HEURIST_IMAGE_MODEL") || "FLUX.1-dev",
                                    width: 1024,
                                    height: 1024,
                                    steps: 30
                                }
                            }
                        }, {
                            type: "GENERATE_IMAGE",
                            payload: {
                                prompt: imagePrompt,
                                model: this.runtime.getSetting("HEURIST_IMAGE_MODEL") || "FLUX.1-dev",
                                width: 1024,
                                height: 1024,
                                steps: 30
                            }
                        });

                        const message = {
                            userId: this.runtime.agentId,
                            roomId: stringToUuid("twitter_image_generation"),
                            agentId: this.runtime.agentId,
                            content: {
                                text: imagePrompt,
                                action: "GENERATE_IMAGE",
                                payload: {
                                    prompt: imagePrompt,
                                    model: this.runtime.getSetting("HEURIST_IMAGE_MODEL") || "FLUX.1-dev",
                                    width: 1024,
                                    height: 1024,
                                    steps: 30
                                }
                            }
                        };

                        const result = await imageAction.handler(
                            this.runtime,
                            message,
                            state,
                            {}
                        );
                        
                        return result;
                    } finally {
                        // Restore the original provider and token
                        this.runtime.character.modelProvider = originalProvider;
                        this.runtime.token = originalToken;
                    }
                } catch (error) {
                    this.endStageMetrics('imageGeneration', false, error as Error);
                    elizaLogger.warn('Continuing without image due to generation failure');
                }
            }

            // Posting stage
            this.tweetGenerationStatus.currentStage = 'posting';
            this.startStageMetrics('posting');
            try {
                if (imageBuffer) {
                    await this.client.twitterClient.sendTweetWithMedia(content, [imageBuffer]);
                    elizaLogger.log("Posted tweet with generated image:", content);
                } else {
                    await this.client.twitterClient.sendTweet(content);
                    elizaLogger.log("Posted tweet:", content);
                }
                this.endStageMetrics('posting', true);
                
                // Update success metrics
                this.tweetGenerationStatus.metrics.overallSuccess = true;
                this.tweetGenerationStatus.metrics.actualPostTime = Date.now();
                this.tweetGenerationStatus.metrics.delay = 
                    this.tweetGenerationStatus.metrics.actualPostTime - 
                    (this.tweetGenerationStatus.metrics.scheduledTime || 0);
                
                this.tweetGenerationStatus.consecutiveFailures = 0;
                this.tweetGenerationStatus.lastSuccessfulPost = Date.now();

                // Log final success metrics
                this.logPerformanceMetrics('Tweet generation completed successfully');
                
            } catch (error) {
                this.endStageMetrics('posting', false, error as Error);
                throw error;
            }

        } catch (error) {
            // Update failure metrics
            this.tweetGenerationStatus.consecutiveFailures++;
            this.tweetGenerationStatus.lastError = error?.message;
            
            // Log comprehensive failure metrics
            this.logPerformanceMetrics(
                `Failed during ${this.tweetGenerationStatus.currentStage} stage`,
                false,
                error as Error
            );

            // Attempt retry if applicable
            if (this.tweetGenerationStatus.retryCount < this.MAX_RETRY_ATTEMPTS) {
                this.tweetGenerationStatus.retryCount++;
                elizaLogger.warn(`Scheduling retry attempt ${this.tweetGenerationStatus.retryCount}/${this.MAX_RETRY_ATTEMPTS}`);
                setTimeout(() => this.generateNewTweet(), this.RETRY_DELAY_MS);
            } else {
                elizaLogger.error('Max retry attempts exceeded, giving up on this tweet generation');
                throw error;
            }
        } finally {
            if (this.tweetGenerationStatus.retryCount >= this.MAX_RETRY_ATTEMPTS) {
                this.tweetGenerationStatus.isGenerating = false;
            }
        }
    }

    /**
     * Post a tweet with one or more images
     * @param text The tweet text
     * @param images Array of image data as Buffer or data URL strings
     * @param replyToTweetId Optional tweet ID to reply to
     */
    async postTweetWithImages(text: string, images: (Buffer | string)[], replyToTweetId?: string) {
        try {
            // Convert any data URLs to Buffers
            const imageBuffers = await Promise.all(images.map(async img => {
                if (Buffer.isBuffer(img)) {
                    return img;
                }
                if (typeof img === 'string' && img.startsWith('data:')) {
                    return await this.imageToBuffer(img);
                }
                throw new Error('Invalid image format. Must be Buffer or data URL string.');
            }));

            await this.client.twitterClient.sendTweetWithMedia(text, imageBuffers, replyToTweetId);
            elizaLogger.log("Posted tweet with custom images:", text);
        } catch (error) {
            elizaLogger.error("Error posting tweet with images:", error);
            throw error;
        }
    }

    private async generateTweetText(): Promise<string | undefined> {
        const startTime = Date.now();
        let stage = 'initialization';
        try {
            elizaLogger.log('Ensuring user exists...');
            stage = 'user_verification';
            await this.runtime.ensureUserExists(
                this.runtime.agentId,
                this.client.profile.username,
                this.runtime.character.name,
                "twitter"
            );

            elizaLogger.log('Fetching timeline data...');
            stage = 'timeline_fetch';
            let homeTimeline: Tweet[] = [];
            let timelineSource = 'cache';

            const cachedTimeline = await this.client.getCachedTimeline();

            if (cachedTimeline) {
                elizaLogger.log('Using cached timeline data');
                homeTimeline = cachedTimeline;
            } else {
                elizaLogger.log('Fetching fresh timeline data');
                timelineSource = 'api';
                homeTimeline = await this.client.fetchHomeTimeline(10);
                await this.client.cacheTimeline(homeTimeline);
            }

            elizaLogger.log('Preparing tweet generation context...');
            stage = 'context_preparation';
            const formattedHomeTimeline =
                `# ${this.runtime.character.name}'s Home Timeline\n\n` +
                homeTimeline
                    .map((tweet) => {
                        return `#${tweet.id}\n${tweet.name} (@${tweet.username})${tweet.inReplyToStatusId ? `\nIn reply to: ${tweet.inReplyToStatusId}` : ""}\n${new Date(tweet.timestamp).toDateString()}\n\n${tweet.text}\n---\n`;
                    })
                    .join("\n");

            const topics = this.runtime.character.topics.join(", ");
            
            elizaLogger.log('Composing tweet generation state...');
            stage = 'state_composition';
            const state = await this.runtime.composeState(
                {
                    userId: this.runtime.agentId,
                    roomId: stringToUuid("twitter_generate_room"),
                    agentId: this.runtime.agentId,
                    content: {
                        text: topics,
                        action: "",
                    },
                },
                {
                    twitterUserName: this.client.profile.username,
                    timeline: formattedHomeTimeline,
                }
            );

            const context = composeContext({
                state,
                template:
                    this.runtime.character.templates?.twitterPostTemplate ||
                    twitterPostTemplate,
            });

            elizaLogger.log('Generating tweet text with AI model...');
            stage = 'ai_generation';
            const generationStart = Date.now();
            const newTweetContent = await generateText({
                runtime: this.runtime,
                context,
                modelClass: ModelClass.SMALL,
            });

            const generationDuration = Math.floor((Date.now() - generationStart) / 1000);
            elizaLogger.log(`AI model response received in ${generationDuration}s`);

            if (!newTweetContent) {
                throw new Error('AI model returned empty tweet content');
            }

            // Replace \n with proper line breaks and trim excess spaces
            stage = 'text_formatting';
            const formattedTweet = newTweetContent
                .replaceAll(/\\n/g, "\n")
                .trim();

            const duration = Math.floor((Date.now() - startTime) / 1000);
            elizaLogger.log(`Successfully generated tweet text in ${duration}s`, {
                timelineSource,
                generationTime: generationDuration,
                totalTime: duration,
                textLength: formattedTweet.length
            });
            
            // Use the helper function to truncate to complete sentence
            return truncateToCompleteSentence(formattedTweet);

        } catch (error) {
            const duration = Math.floor((Date.now() - startTime) / 1000);
            elizaLogger.error('Error generating tweet text:', {
                error: error?.message,
                stack: error?.stack,
                stage,
                duration: `${duration}s`,
                failurePoint: stage,
                attempts: this.tweetGenerationStatus.attempts,
                retryCount: this.tweetGenerationStatus.retryCount,
                consecutiveFailures: this.tweetGenerationStatus.consecutiveFailures
            });
            return undefined;
        }
    }
}
