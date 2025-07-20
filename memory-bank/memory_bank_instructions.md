# Memory Bank Instructions
## LIGO CPC+SNN Gravitational Wave Detection Pipeline

### Cel Memory Bank
Ten Memory Bank stanowi centralną bazę wiedzy dla projektu detekcji fal grawitacyjnych wykorzystujący Contrastive Predictive Coding + Spiking Neural Networks. Zapewnia spójność w rozwoju, ułatwia onboarding nowych współpracowników i dokumentuje kluczowe decyzje architektoniczne.

### Struktura i Zastosowanie

#### Core Files Overview
1. **projectbrief.md** - Fundament projektu, misja i cele techniczne
2. **productContext.md** - Analiza problemu, competitive landscape, target users
3. **activeContext.md** - Bieżący stan rozwoju, kolejne kroki, blockers
4. **systemPatterns.md** - Architektura systemu, wzorce projektowe, API design
5. **techContext.md** - Stack technologiczny, setup środowiska, constraints
6. **progress.md** - Historia postępów, completed milestones, remaining work
7. **memory_bank_instructions.md** - Ten plik z instrukcjami

### Workflow Guidelines

#### Dla AI Assistant
1. **Przed rozpoczęciem pracy** - przeczytaj activeContext.md i progress.md
2. **Podczas implementacji** - odnosij się do systemPatterns.md dla spójności architektury
3. **Po major changes** - zaktualizuj odpowiednie pliki Memory Bank
4. **Przy problemach** - sprawdź techContext.md dla known issues i solutions

#### Dla Deweloperów
1. **Onboarding** - zacznij od projectbrief.md → productContext.md → techContext.md
2. **Daily work** - sprawdź activeContext.md dla current priorities
3. **Architecture decisions** - dokumentuj w systemPatterns.md
4. **Completed features** - aktualizuj progress.md

### Update Protocol

#### Kiedy Aktualizować
- **activeContext.md**: każdy sprint/major milestone
- **systemPatterns.md**: nowe komponenty, zmiana API
- **techContext.md**: upgrade dependencies, nowe tools
- **progress.md**: completed features, test results
- **projectbrief.md**: zmiana scope lub success metrics
- **productContext.md**: nowa competitive intelligence

#### Jak Aktualizować
1. **Atomic updates** - jedna zmiana per commit
2. **Cross-reference** - łącz związane informacje między plikami
3. **Versioning** - używaj dat dla major milestones
4. **Consistency** - terminologia musi być spójna w całym Memory Bank

### Knowledge Categories

#### Technical Knowledge
- **Architecture patterns** → systemPatterns.md
- **Implementation details** → techContext.md + progress.md
- **Performance metrics** → progress.md
- **Known issues/solutions** → techContext.md

#### Product Knowledge
- **User needs** → productContext.md
- **Market analysis** → productContext.md
- **Success metrics** → projectbrief.md
- **Roadmap** → activeContext.md

#### Project Knowledge
- **Current status** → activeContext.md
- **Completed work** → progress.md
- **Next priorities** → activeContext.md
- **Team decisions** → systemPatterns.md

### Best Practices

#### Writing Guidelines
1. **Clarity** - używaj precyzyjnej terminologii
2. **Brevity** - koncentruj się na key insights
3. **Context** - zawsze podawaj dlaczego, nie tylko co
4. **Examples** - konkretne przykłady code/config
5. **Links** - odnośniki do external resources

#### Maintenance Rules
1. **Regular reviews** - tygodniowo sprawdź activeContext.md
2. **Archive old info** - przenoś outdated content do progress.md
3. **Validate links** - sprawdź external references
4. **Sync with code** - Memory Bank musi odzwierciedlać aktualny kod

### Integration z Development Workflow

#### Pre-commit Checklist
- [ ] Czy nowe componenty są udokumentowane w systemPatterns.md?
- [ ] Czy activeContext.md odzwierciedla current state?
- [ ] Czy major features są recorded w progress.md?

#### Sprint Planning
1. Review activeContext.md dla priorities
2. Check progress.md dla blocked items
3. Validate systemPatterns.md dla architecture constraints
4. Update activeContext.md z nowym planem

#### Code Reviews
- Sprawdź zgodność z patterns z systemPatterns.md
- Verify tech choices zgodnie z techContext.md
- Ensure documentation updates w relevant files

### Emergency Procedures

#### Critical Issues
1. Document problem w techContext.md (known issues section)
2. Update activeContext.md z impact assessment
3. Record resolution w progress.md po fix

#### Major Architecture Changes
1. Discuss w kontekście systemPatterns.md
2. Update all affected Memory Bank files
3. Record rationale i alternatives considered

## Current Priority Areas - PROJECT STATUS UPDATE

### Foundation Phase (Week 1-2) ✅ COMPLETED AHEAD OF SCHEDULE - 2025-01-06
- ✅ Environment setup completely operational (JAX + Metal + Spyx + GWOSC)
- ✅ Complete architecture implemented and verified
- ✅ GWOSC data access working with GWpy integration  
- ✅ All core module implementations tested and functional
- ✅ Memory Bank structure established with complete documentation

### Data Pipeline Phase (Week 3-4) 🚀 CURRENT ACTIVE PHASE
- **Real GWOSC data processing** with quality validation pipeline
- **CPC encoder InfoNCE training** implementation and optimization
- **Production preprocessing pipeline** with performance benchmarks
- **Apple Silicon optimization** for training and inference workflows

### Historic Achievement Status ✅ BREAKTHROUGH ACCOMPLISHED
**FIRST IN WORLD**: Neuromorphic gravitational wave detection environment operational on Apple Silicon

### Memory Bank Update Protocol for Data Pipeline Phase
1. **Daily**: Update activeContext.md with progress and blockers
2. **Weekly**: Document training results and performance metrics in progress.md  
3. **Per milestone**: Update systemPatterns.md with new implementations
4. **Critical decisions**: Record rationale in appropriate context files 