
╭─── Claude Code v2.0.69 ─────────────────────────────────────────────────────────────╮
│                                                    │ Recent activity                │
│              Welcome back CloudyCadet!             │ No recent activity             │
│                                                    │ ────────────────────────────── │
│                     * ▐▛███▜▌ *                    │ 3 guest passes                 │
│                    * ▝▜█████▛▘ *                   │                                │
│                     *  ▘▘ ▝▝  *                    │ [✻] [✻] [✻]                    │
│                                                    │                                │
│    Opus 4.5 · Claude Max · mmerc808@gmail.com's    │ Share Claude Code with friends │
│    Organization                                    │ /passes                        │
│                   C:\code\islands                  │                                │
╰─────────────────────────────────────────────────────────────────────────────────────╯

> i need a proposed directory structure. a comprehensive list of classes and functions is needed. 

● I'll explore the codebase to understand its current structure and document all classes and functions.

● Explore(Explore codebase structure and catalog classes/functions)
  ⎿  Done (49 tool uses · 62.2k tokens · 4m 6s)

● Proposed Directory Structure

  islands/
  ├── matts/                          # Core serializable context library
  │   ├── __init__.py                 # Library initialization, exports
  │   ├── signal_system.py            # Signal lines, observers, circuit states
  │   ├── context_system.py           # SerializableExecutionContext, chains
  │   ├── context_serialization.py    # Compressed context transmission
  │   ├── live_code_system.py         # Runtime code serialization
  │   ├── generator_system.py         # Generator composition patterns
  │   ├── graph_system.py             # Basic relationship graphs
  │   ├── functional_interface.py     # Primary API functions
  │   └── examples/
  │       ├── __init__.py
  │       ├── ai_collaboration_system.py
  │       ├── distributed_workers_system.py
  │       ├── game_narrative_system.py
  │       ├── usage_patterns.py
  │       └── demo_system.py
  │
  ├── signbook/                       # AI persistence protocol
  │   ├── sonar-sign/
  │   │   └── signbook.py
  │   └── last-sign/
  │       └── signbook_mcp.py
  │
  ├── Mechanics/                      # Advanced narrative mechanics
  │   ├── hero_quest_chronicle.py
  │   ├── traversing_spirit_stick.py
  │   ├── spirit_stick.py
  │   └── stream_processor.py
  │
  ├── files(8)/                       # Graph and narrative utilities
  │   ├── graph_core.py
  │   ├── graph_walker.py
  │   ├── callback_patterns.py
  │   └── narrative.py
  │
  ├── inage-text/                     # Infocom-style interactive fiction
  │   └── files/
  │       └── my_infocom_entity_system.py
  │
  ├── islands/                        # Island archipelago simulation
  │   └── main.py
  │
  ├── infocom-transfer/               # Infocom content transfer
  ├── snapshots/                      # Context snapshots storage
  ├── docs/                           # Documentation
  ├── _old/                           # Archived code
  │
  ├── # Top-level applications
  ├── story_world_pyside6.py          # Story world engine with OpenGL
  ├── proc_streamer_v1_6.py           # Ollama chat interface
  ├── proc_streamer_legacy_ui.py      # Legacy chat UI
  ├── aqua.py                         # Advanced word processor
  ├── simple_word_processor.py        # Basic word processor
  ├── unified_console_tab.py          # Console tab widget
  ├── persistent_context_iterator.py  # Context iteration system
  ├── narrative_chain_iterator_system.py
  ├── runtime_hotswap_system.py       # Bytecode hot-swapping
  ├── live_code_system.py             # Code serialization (copy)
  ├── context_serialization.py        # Context serialization (copy)
  ├── english_language_code_object.py # NLP rule chains
  ├── code_object_utility.py          # Python process analyzer
  ├── thin-wrapper-patterns.py        # Wrapper design patterns
  │
  ├── CLAUDE.md
  ├── settings.json
  └── env/                            # Virtual environment

  ---
  Comprehensive Class & Function Inventory

  matts/signal_system.py

  | Type     | Name               | Line |
  |----------|--------------------|------|
  | Enum     | SignalPriority     | 50   |
  | Enum     | SignalType         | 59   |
  | Enum     | ObserverPriority   | 92   |
  | Enum     | CircuitState       | 99   |
  | Class    | SignalPayload      | 107  |
  | Class    | ObserverStats      | 150  |
  | Class    | Observer (ABC)     | 174  |
  | Class    | CallbackObserver   | 267  |
  | Class    | CircuitBreaker     | 292  |
  | Class    | ObserverWorkerPool | 388  |
  | Class    | PriorityDispatcher | 422  |
  | Class    | SignalLine         | 543  |
  | Class    | ExampleObserver    | 781  |
  | Function | signal_handler()   | 755  |

  matts/context_system.py

  | Type  | Name                         | Line |
  |-------|------------------------------|------|
  | Enum  | ContextState                 | 48   |
  | Class | DependencyBundler            | 66   |
  | Class | ContextObserver (ABC)        | 148  |
  | Class | SignalAwareContextObserver   | 164  |
  | Class | CompositeObserver            | 200  |
  | Class | CallbackIteratorObserver     | 233  |
  | Class | ContextSnapshot              | 285  |
  | Class | CircularReferenceDetector    | 431  |
  | Class | ContextChainNode             | 467  |
  | Class | SerializableExecutionContext | 686  |
  | Class | ContextGarbageCollector      | 902  |
  | Class | SerializableContextLibrary   | 997  |

  matts/context_serialization.py

  | Type  | Name                                        | Line |
  |-------|---------------------------------------------|------|
  | Class | SerializedContextMetadata                   | 47   |
  | Class | Signal                                      | 87   |
  | Class | ContextAwareSignalObserver                  | 161  |
  | Class | FastDependencyBundler                       | 197  |
  | Class | OptimizedSerializer                         | 316  |
  | Class | HighPerformanceSignalBus                    | 402  |
  | Class | SerializableExecutionContextWithPortability | 495  |

  matts/live_code_system.py

  | Type      | Name                     | Line |
  |-----------|--------------------------|------|
  | Enum      | CodeSerializationMethod  | 47   |
  | Class     | SerializedSourceCode     | 57   |
  | Class     | CompleteSerializedCode   | 120  |
  | Class     | SourceCodeSerializer     | 177  |
  | Class     | ContextAwareDeserializer | 426  |
  | Class     | BytecodeExecutionEngine  | 610  |
  | Class     | RuntimeSourceEditor      | 735  |
  | Class     | RuntimeCodeCache         | 863  |
  | Class     | CompleteLiveCodeSystem   | 981  |
  | Class     | LiveCodeCallbackSystem   | 1041 |
  | Exception | SecurityError            | 1102 |
  | Exception | DeserializationError     | 1106 |

  matts/generator_system.py

  | Type     | Name                                   | Line |
  |----------|----------------------------------------|------|
  | Enum     | GeneratorCompositionPattern            | 24   |
  | Class    | GeneratorBranch                        | 35   |
  | Class    | GeneratorStateBranch                   | 69   |
  | Class    | GeneratorCompositionEngine             | 102  |
  | Class    | AdvancedGeneratorComposer              | 211  |
  | Function | create_data_generator_factory()        | 480  |
  | Function | create_transformer_generator_factory() | 498  |
  | Function | create_filter_generator_factory()      | 528  |
  | Function | create_aggregator_generator_factory()  | 561  |
  | Function | demo_generator_composition()           | 608  |
  | Function | demo_pipeline_processing()             | 724  |

  matts/graph_system.py

  | Type  | Name                   | Line |
  |-------|------------------------|------|
  | Enum  | RelationshipType       | 20   |
  | Class | GraphNode              | 32   |
  | Class | GraphEdge              | 41   |
  | Class | BasicRelationshipGraph | 51   |

  matts/functional_interface.py

  | Type     | Name                              | Line |
  |----------|-----------------------------------|------|
  | Function | initialize_library()              | 24   |
  | Function | initialize_signal_line()          | 31   |
  | Function | create_context()                  | 43   |
  | Function | get_context()                     | 50   |
  | Function | destroy_context()                 | 55   |
  | Function | bind_callback_with_dependencies() | 60   |
  | Function | create_context_snapshot()         | 68   |
  | Function | serialize_context()               | 72   |
  | Function | deserialize_context()             | 76   |
  | Function | emit_signal()                     | 84   |
  | Function | transmit_context()                | 123  |
  | Function | register_signal_callback()        | 135  |
  | Function | run_garbage_collection()          | 153  |
  | Function | get_performance_stats()           | 158  |
  | Function | get_signal_stats()                | 170  |
  | Function | create_generator_composer()       | 182  |
  | Function | register_generator_factory()      | 190  |
  | Function | add_relationship()                | 200  |
  | Function | find_related()                    | 221  |
  | Function | get_relationship_path()           | 233  |
  | Function | create_live_code_system()         | 246  |
  | Function | create_callback_system()          | 251  |
  | Function | serialize_source_code()           | 256  |
  | Function | serialize_function()              | 263  |
  | Function | pause_callback_chain()            | 273  |
  | Function | resume_callback_chain()           | 284  |
  | Function | add_context_observer()            | 293  |
  | Function | hot_swap_context_version()        | 302  |
  | Function | update_context_data()             | 315  |
  | Function | get_context_data()                | 324  |
  | Function | get_context_snapshots()           | 334  |
  | Function | create_multiple_contexts()        | 348  |
  | Function | transmit_to_multiple_contexts()   | 356  |
  | Function | batch_hot_swap()                  | 371  |
  | Function | get_detailed_context_info()       | 390  |
  | Function | list_all_contexts()               | 420  |
  | Function | health_check()                    | 436  |
  | Function | quick_start()                     | 474  |
  | Function | demo_setup()                      | 482  |

  matts/examples/ai_collaboration_system.py

  | Type     | Name                           | Line |
  |----------|--------------------------------|------|
  | Enum     | AgentRole                      | 38   |
  | Enum     | CollaborationMode              | 48   |
  | Enum     | TaskComplexity                 | 56   |
  | Class    | AIAgent                        | 64   |
  | Class    | CollaborativeTask              | 79   |
  | Class    | ReasoningStep                  | 98   |
  | Class    | CollaborationExchange          | 111  |
  | Class    | AICollaborationSystem          | 126  |
  | Function | demo_ai_collaboration_system() | 1290 |

  matts/examples/distributed_workers_system.py

  | Type     | Name                             | Line |
  |----------|----------------------------------|------|
  | Enum     | WorkerStatus                     | 36   |
  | Enum     | TaskPriority                     | 44   |
  | Class    | WorkerNode                       | 53   |
  | Class    | DistributedTask                  | 67   |
  | Class    | TaskResult                       | 86   |
  | Class    | DistributedWorkerSystem          | 101  |
  | Function | execute_task()                   | 477  |
  | Function | demo_distributed_worker_system() | 729  |
  | Function | process_data()                   | 757  |
  | Function | train_model()                    | 782  |

  matts/examples/game_narrative_system.py

  | Type  | Name                | Line |
  |-------|---------------------|------|
  | Enum  | GameEventType       | 35   |
  | Enum  | QuestStatus         | 45   |
  | Class | GameCharacter       | 53   |
  | Class | Quest               | 65   |
  | Class | GameEvent           | 76   |
  | Class | GameNarrativeSystem | 90   |

  ---
  signbook/sonar-sign/signbook.py

  | Type     | Name                 | Line |
  |----------|----------------------|------|
  | Class    | Entry                | 109  |
  | Class    | Signbook             | 187  |
  | Function | generate_signature() | 38   |
  | Function | parse_signature()    | 74   |
  | Function | verify_signature()   | 92   |
  | Function | create_signbook()    | 316  |
  | Function | quick_sign()         | 346  |

  signbook/last-sign/signbook_mcp.py

  | Type     | Name                   | Line |
  |----------|------------------------|------|
  | Class    | MCPServer              | 432  |
  | Function | generate_signature()   | 90   |
  | Function | compute_checksum()     | 113  |
  | Function | read_signbook_md()     | 126  |
  | Function | read_signatures_json() | 133  |
  | Function | count_md_entries()     | 143  |
  | Function | add_entry_to_md()      | 153  |
  | Function | add_entry_to_json()    | 227  |
  | Function | git_status()           | 295  |
  | Function | git_push()             | 354  |
  | Function | cli_mode()             | 769  |

  ---
  story_world_pyside6.py

  | Type     | Name                  | Line |
  |----------|-----------------------|------|
  | Enum     | EntityType            | 50   |
  | Enum     | MessageType           | 73   |
  | Class    | EntityData            | 58   |
  | Class    | LogMessage            | 82   |
  | Class    | NLPProcessor          | 88   |
  | Class    | EntityToken           | 148  |
  | Class    | OpenGLFontRenderer    | 226  |
  | Class    | EntityTokenWidget     | 511  |
  | Class    | EntityContainerWidget | 631  |
  | Class    | SceneGraphWidget      | 662  |
  | Class    | MessageLogWidget      | 711  |
  | Class    | StoryWorldEngine      | 751  |
  | Function | main()                | 1394 |

  proc_streamer_v1_6.py

  | Type     | Name              | Line |
  |----------|-------------------|------|
  | Class    | Theme             | 59   |
  | Class    | ThemeManager      | 77   |
  | Class    | GlobalSettings    | 120  |
  | Class    | Snapshot          | 164  |
  | Class    | DocSession        | 170  |
  | Class    | AssistChannel     | 193  |
  | Class    | LayeredStyleMixin | 330  |
  | Class    | UnifiedConsoleTab | 376  |
  | Class    | SimpleTextEditor  | 529  |
  | Class    | AssistantPanel    | 546  |
  | Class    | MainWindow        | 582  |
  | Function | main()            | 745  |

  runtime_hotswap_system.py

  | Type     | Name                                | Line |
  |----------|-------------------------------------|------|
  | Class    | BytecodeInstruction                 | 39   |
  | Class    | BytecodeEditor                      | 53   |
  | Class    | RuntimeFunctionRegistry             | 200  |
  | Class    | EntitySystem                        | 342  |
  | Class    | DecoratorHotSwapSystem              | 471  |
  | Class    | CodeDistributionSignal              | 548  |
  | Class    | NetworkCodeDistributor              | 563  |
  | Class    | LiveUpdateSignalHandler             | 1120 |
  | Function | demo_runtime_hot_swapping()         | 689  |
  | Function | ultimate_player_onDamage()          | 915  |
  | Function | game_scoring_function()             | 974  |
  | Function | create_live_update_enabled_system() | 1213 |

  persistent_context_iterator.py

  | Type     | Name                      | Line |
  |----------|---------------------------|------|
  | Enum     | OperationType             | 38   |
  | Class    | ContextOperation          | 50   |
  | Class    | ContextWindow             | 85   |
  | Class    | PersistentContextIterator | 322  |
  | Class    | ScheduledTask             | 548  |
  | Class    | TaskScheduler             | 576  |
  | Class    | GameLoopIntegrator        | 872  |
  | Function | create_demo_tasks()       | 1119 |
  | Function | simulate_game_events()    | 1208 |
  | Function | demonstrate_system()      | 1234 |

  narrative_chain_iterator_system.py

  | Type     | Name                    | Line |
  |----------|-------------------------|------|
  | Class    | ProcessControlManager   | 42   |
  | Enum     | ChainEventType          | 200  |
  | Class    | ChainEvent              | 212  |
  | Class    | ChainLink               | 236  |
  | Class    | NarrativeChain          | 283  |
  | Class    | CrossChainHandler       | 362  |
  | Class    | ChainIteratorEventQueue | 500  |
  | Class    | StoryExecutionThread    | 773  |
  | Function | create_story_example()  | 827  |

  english_language_code_object.py

  | Type     | Name                              | Line |
  |----------|-----------------------------------|------|
  | Enum     | LanguageRuleType                  | 19   |
  | Enum     | ChainDecisionType                 | 30   |
  | Class    | LanguageToken                     | 42   |
  | Class    | RuleChain                         | 56   |
  | Class    | ChainNode                         | 108  |
  | Class    | EnglishLanguageCodeObject         | 175  |
  | Class    | AdaptiveChainLearner              | 687  |
  | Class    | SelfModifyingLanguageSystem       | 928  |
  | Function | demonstrate_english_code_object() | 615  |
  | Function | demonstrate_adaptive_evolution()  | 1092 |

  code_object_utility.py

  | Type     | Name                               | Line |
  |----------|------------------------------------|------|
  | Enum     | CodeObjectType                     | 40   |
  | Class    | ProcessInfo                        | 119  |
  | Class    | ThreadInfo                         | 139  |
  | Class    | MemoryInfo                         | 164  |
  | Class    | ModuleInfo                         | 178  |
  | Class    | FrameInfo                          | 203  |
  | Class    | ParameterInfo                      | 238  |
  | Class    | SignatureInfo                      | 255  |
  | Class    | ASTMetadata                        | 268  |
  | Class    | BytecodeInfo                       | 318  |
  | Class    | ComprehensiveCodeObjectMetadata    | 347  |
  | Class    | CodeObjectContext (ABC)            | 467  |
  | Class    | ComprehensivePythonProcessAnalyzer | 486  |
  | Function | analyze_comprehensive()            | 1305 |
  | Function | get_full_process_snapshot()        | 1320 |

  ---
  Mechanics/hero_quest_chronicle.py

  | Type     | Name                        | Line |
  |----------|-----------------------------|------|
  | Enum     | MessageType                 | 70   |
  | Class    | Message                     | 95   |
  | Class    | Chronicle                   | 155  |
  | Class    | MessagingInterface (ABC)    | 488  |
  | Class    | StdinInterface              | 509  |
  | Class    | QueueInterface              | 525  |
  | Class    | QuestState                  | 549  |
  | Class    | HeroActor                   | 571  |
  | Class    | QuestNarrator               | 773  |
  | Function | create_console_logger()     | 837  |
  | Function | create_file_logger()        | 846  |
  | Function | create_buffer_logger()      | 855  |
  | Function | example_simulated_quest()   | 866  |
  | Function | example_interactive_quest() | 921  |

  Mechanics/traversing_spirit_stick.py

  | Type     | Name                           | Line |
  |----------|--------------------------------|------|
  | Protocol | Traversable[T]                 | 45   |
  | Protocol | HasNext[T]                     | 62   |
  | Protocol | HasChildren[T]                 | 77   |
  | Class    | Branch[T]                      | 91   |
  | Class    | BranchFinder[T]                | 267  |
  | Class    | Prediction                     | 355  |
  | Class    | Prose                          | 401  |
  | Class    | ChronicleEntry                 | 424  |
  | Class    | Chronicle                      | 457  |
  | Class    | ProseOracle                    | 536  |
  | Class    | ChronicleOracle                | 595  |
  | Class    | DualOracle                     | 634  |
  | Class    | SpiritStick[T]                 | 678  |
  | Class    | Resolver                       | 848  |
  | Class    | StoryNode                      | 934  |
  | Function | demo_traversing_spirit_stick() | 961  |
  | Function | demo_linked_list_branch()      | 1034 |
  | Function | demo_conditional_branching()   | 1054 |

  ---
  files(8)/graph_core.py

  | Type     | Name               | Line |
  |----------|--------------------|------|
  | Enum     | NodeType           | 28   |
  | Enum     | EdgeType           | 42   |
  | Class    | GraphNode          | 71   |
  | Class    | Edge               | 158  |
  | Class    | IslandGraph        | 204  |
  | Function | create_location()  | 610  |
  | Function | create_item()      | 622  |
  | Function | create_character() | 635  |
  | Function | create_prose()     | 648  |

  files(8)/graph_walker.py

  | Type     | Name                | Line |
  |----------|---------------------|------|
  | Enum     | TraversalStrategy   | 34   |
  | Class    | WalkerCallback      | 48   |
  | Class    | GraftBranch         | 86   |
  | Class    | WalkerContext       | 118  |
  | Class    | GraphWalker         | 188  |
  | Function | print_description() | 796  |
  | Function | collect_items()     | 802  |
  | Function | reveal_secrets()    | 809  |

  files(8)/narrative.py

  | Type     | Name               | Line |
  |----------|--------------------|------|
  | Enum     | ProseType          | 32   |
  | Enum     | StoryState         | 44   |
  | Class    | ProseFragment      | 58   |
  | Class    | StoryBeat          | 147  |
  | Class    | ProseRegistry      | 186  |
  | Class    | StoryManager       | 301  |
  | Class    | NarrativeWalker    | 477  |
  | Function | make_description() | 613  |
  | Function | make_dialogue()    | 622  |
  | Function | make_narration()   | 631  |
  | Function | make_beat()        | 640  |

  ---
  inage-text/files/my_infocom_entity_system.py

  | Type     | Name                   | Line |
  |----------|------------------------|------|
  | Enum     | EntityType             | 30   |
  | Enum     | EntityState            | 40   |
  | Enum     | LinkType               | 50   |
  | Enum     | RelationalKind         | 57   |
  | Enum     | LogicalKind            | 79   |
  | Enum     | FragmentCategory       | 89   |
  | Class    | FragmentCondition      | 109  |
  | Class    | ProseFragment          | 127  |
  | Class    | ProseCompositor        | 169  |
  | Class    | LinkCondition          | 251  |
  | Class    | Link                   | 264  |
  | Class    | Node (ABC)             | 297  |
  | Class    | Vocabulary             | 362  |
  | Class    | Entity                 | 378  |
  | Class    | EntityGraph            | 482  |
  | Class    | Action                 | 673  |
  | Class    | ActionProcessor        | 686  |
  | Function | entity_from_pipeline() | 778  |
  | Function | serialize_graph()      | 873  |
  | Function | create_demo_world()    | 928  |

  ---
  Summary

  | Category       | Classes | Functions |
  |----------------|---------|-----------|
  | matts/ core    | 45      | 55        |
  | matts/examples | 15      | 8         |
  | signbook/      | 3       | 12        |
  | Top-level apps | 55      | 20        |
  | Mechanics/     | 25      | 12        |
  | files(8)/      | 15      | 25        |
  | inage-text/    | 15      | 5         |
  | Total          | ~175    | ~140      |
