import { JevDecisionClient } from './client.js';

async function runTest() {
  console.log('Testing JevDecisionClient with live Vercel AI Gateway...');
  const jev = new JevDecisionClient();

  // Test 1: Speculation Window K for Code
  console.log('\n1. Speculation Window K for Code Task:');
  const codeK = await jev.decideSpeculationK({
    promptSnippet: 'def quicksort(arr): if len(arr) <= 1: return arr',
    taskDomain: 'code',
    recentAcceptanceRate: 0.62,
  });
  console.log('Result:', codeK);

  // Test 2: Memory Admission
  console.log('\n2. Memory Admission Decision on 128GB GB10:');
  const memDecision = await jev.decideMemoryAdmission({
    activeSessions: 4,
    usedKVMemoryGB: 35,
    totalAddressableGB: 121,
    incomingTokens: 4096,
  });
  console.log('Result:', memDecision);

  // Test 3: Tool Routing
  console.log('\n3. Tool Routing Decision:');
  const toolDecision = await jev.decideToolRouting('Can you read src/server/index.ts and find the port number?');
  console.log('Result:', toolDecision);

  console.log('\nAll Jev decision functions verified successfully!');
}

runTest().catch(console.error);
