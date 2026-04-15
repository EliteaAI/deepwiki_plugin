/**
 * Example usage of the EliteA Sandbox Client
 * Demonstrates tool testing with sync and async patterns
 */

import { SandboxClient } from './sandbox_client.js';

// Configuration
const config = {
    baseUrl: process.env.ELITEA_BASE_URL || 'https://elitea.example.com',
    projectId: parseInt(process.env.ELITEA_PROJECT_ID || '1'),
    authToken: process.env.ELITEA_AUTH_TOKEN || 'your-token-here',
    XSECRET: process.env.ELITEA_SECRET || 'secret'
};

async function exampleSyncToolTest() {
    console.log('\n=== Synchronous Tool Test ===\n');
    
    const client = new SandboxClient(config);
    
    try {
        // Test a tool synchronously - response waits for completion
        const result = await client.testToolSync(123, {
            tool: 'example_tool',
            testing_name: 'sync_test_run',
            input: {
                text: 'Hello, World!',
                options: {
                    uppercase: true,
                    reverse: false
                }
            },
            output: {},
            user_input: 'Process this text',
            call_type: 'tool'
        });
        
        console.log('Tool execution result:', result);
        return result;
        
    } catch (error) {
        console.error('Sync test failed:', error.message);
        throw error;
    }
}

async function exampleAsyncToolTest() {
    console.log('\n=== Asynchronous Tool Test with Task Tracking ===\n');
    
    const client = new SandboxClient(config);
    
    try {
        // Start async tool test
        // Note: sid is optional - only needed for Socket.IO streaming
        // If not provided, use polling to get results
        
        const asyncResult = await client.testToolAsync(123, {
            tool: 'long_running_tool',
            testing_name: 'async_test_run',
            input: {
                dataset: Array.from({ length: 1000 }, (_, i) => i),
                processing_type: 'complex'
            },
            user_input: 'Process large dataset',
            call_type: 'tool'
        });
        
        const taskId = asyncResult.task_id;
        console.log('Task started with ID:', taskId);
        
        // Wait for completion with polling
        const finalResult = await client.waitForTask(taskId, 300);
        console.log('Task completed:', finalResult);
        
        return finalResult;
        
    } catch (error) {
        console.error('Async test failed:', error.message);
        throw error;
    }
}

async function exampleAsyncWithProgressTracking() {
    console.log('\n=== Async Tool Test with Progress Tracking ===\n');
    
    const client = new SandboxClient(config);
    
    try {
        // Start the async task (without sid - we'll poll instead)
        const asyncResult = await client.testToolAsync(456, {
            tool: 'data_analyzer',
            testing_name: 'progress_tracked_test',
            input: { data_size: 'large' },
            user_input: 'Analyze data with progress'
        });
        
        const taskId = asyncResult.task_id;
        console.log('Task ID:', taskId);
        
        // Poll with progress callback
        const result = await client.pollTaskStatus(
            taskId,
            (progress) => {
                console.log(`[Attempt ${progress.attempts}] Status: ${progress.status}`);
                if (progress.meta) {
                    console.log('  Meta:', JSON.stringify(progress.meta, null, 2));
                }
            },
            120,  // max 120 attempts
            1000  // check every 1 second
        );
        
        console.log('\nFinal result:', result);
        return result;
        
    } catch (error) {
        console.error('Progress tracking test failed:', error.message);
        throw error;
    }
}

async function exampleManualTaskMonitoring() {
    console.log('\n=== Manual Task Status Monitoring ===\n');
    
    const client = new SandboxClient(config);
    
    try {
        // Start async task (no sid needed for polling pattern)
        const asyncResult = await client.testToolAsync(789, {
            tool: 'report_generator',
            testing_name: 'manual_monitoring_test',
            input: { report_type: 'detailed' },
            user_input: 'Generate comprehensive report'
        });
        
        const taskId = asyncResult.task_id;
        console.log('Task started:', taskId);
        
        // Manual monitoring loop
        let isComplete = false;
        let attempts = 0;
        const maxAttempts = 60;
        
        while (!isComplete && attempts < maxAttempts) {
            attempts++;
            
            // Wait a bit
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Check status
            const status = await client.getTaskStatus(taskId, true, false);
            console.log(`Attempt ${attempts}: ${status.status}`);
            
            if (!['PENDING', 'STARTED', 'RETRY'].includes(status.status)) {
                isComplete = true;
                
                // Get final result
                const finalStatus = await client.getTaskStatus(taskId, true, true);
                
                if (finalStatus.status === 'SUCCESS') {
                    console.log('\nTask succeeded!');
                    console.log('Result:', finalStatus.result);
                    return finalStatus.result;
                } else {
                    console.error('\nTask failed with status:', finalStatus.status);
                    console.error('Error:', finalStatus.result);
                    throw new Error('Task failed');
                }
            }
        }
        
        if (!isComplete) {
            throw new Error('Task monitoring timeout');
        }
        
    } catch (error) {
        console.error('Manual monitoring failed:', error.message);
        throw error;
    }
}

async function exampleStoppingTask() {
    console.log('\n=== Task Cancellation Example ===\n');
    
    const client = new SandboxClient(config);
    
    try {
        // Start a long-running task (no sid needed)
        const asyncResult = await client.testToolAsync(999, {
            tool: 'slow_processor',
            testing_name: 'cancellation_test',
            input: { duration: 'very_long' },
            user_input: 'Start long process'
        });
        
        const taskId = asyncResult.task_id;
        console.log('Task started:', taskId);
        
        // Simulate checking status a few times
        for (let i = 0; i < 3; i++) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            const status = await client.getTaskStatus(taskId);
            console.log(`Status check ${i + 1}: ${status.status}`);
        }
        
        // Decide to cancel
        console.log('\nCancelling task...');
        await client.stopTaskById(taskId);
        console.log('Task cancelled successfully');
        
        // Verify it stopped
        const finalStatus = await client.getTaskStatus(taskId);
        console.log('Final status:', finalStatus.status);
        
    } catch (error) {
        console.error('Task cancellation example failed:', error.message);
    }
}

async function exampleArtifactWorkflow() {
    console.log('\n=== Artifact Management with Tool Results ===\n');
    
    const client = new SandboxClient(config);
    
    try {
        // Create artifact manager
        const artifacts = await client.artifact('tool-results');
        
        // Run a tool test
        const result = await client.testToolSync(123, {
            tool: 'data_processor',
            testing_name: 'artifact_test',
            input: { data: [1, 2, 3, 4, 5] },
            user_input: 'Process and save'
        });
        
        // Save result to artifact
        const resultJson = JSON.stringify(result, null, 2);
        await artifacts.create('tool_result.json', resultJson);
        console.log('Result saved to artifact');
        
        // List artifacts
        const artifactList = await artifacts.list(null, false);
        console.log('Artifacts in bucket:', artifactList);
        
        // Read it back
        const savedResult = await artifacts.get('tool_result.json');
        console.log('Retrieved result:', savedResult);
        
        return result;
        
    } catch (error) {
        console.error('Artifact workflow failed:', error.message);
        throw error;
    }
}

// Main execution
async function main() {
    console.log('EliteA Sandbox Client - Tool Testing Examples');
    console.log('============================================');
    
    try {
        // Run examples
        await exampleSyncToolTest();
        await exampleAsyncToolTest();
        await exampleAsyncWithProgressTracking();
        await exampleManualTaskMonitoring();
        await exampleStoppingTask();
        await exampleArtifactWorkflow();
        
        console.log('\n✅ All examples completed successfully!');
        
    } catch (error) {
        console.error('\n❌ Examples failed:', error.message);
        process.exit(1);
    }
}

// Export for use as module
export {
    exampleSyncToolTest,
    exampleAsyncToolTest,
    exampleAsyncWithProgressTracking,
    exampleManualTaskMonitoring,
    exampleStoppingTask,
    exampleArtifactWorkflow
};

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
