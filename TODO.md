# TODO - Refresh Token Rotation (Enterprise, Replay Detection)

## Plan checkpoints
- [ ] Step 1: Add DB models for refresh token families + refresh tokens (hashed-only storage, indexes)


- [ ] Step 2: Add JWT utilities (access/refresh signing, claims, TTL)

- [ ] Step 3: Add refresh rotation service with one-time-use enforcement + replay detection + family revocation
- [ ] Step 4: Add audit logging helper (structured logs; never log raw tokens)
- [ ] Step 5: Add API endpoints: login (JWT issuance), refresh, logout
- [ ] Step 6: Add concurrency-safe rotation logic (transaction / conditional update)
- [ ] Step 7: Add tests: unit + integration + concurrency + reuse detection + family revocation
- [ ] Step 8: Update docs/security notes
- [ ] Step 9: Run full test suite + ensure CI compatibility

