//go:build integration

package zvec

import "testing"

// addVectorField adds a VECTOR_FP32 field to cs, optionally with an HNSW index.
func addVectorField(t *testing.T, cs *CollectionSchema, name string, dim uint32, hnsw bool) {
	t.Helper()
	fs := NewFieldSchema(name, DataTypeVectorFP32, false, dim)
	if fs == nil {
		t.Fatalf("NewFieldSchema(%q) returned nil", name)
	}
	if hnsw {
		ip, err := NewHNSWIndexParams(MetricTypeCosine, 16, 200)
		if err != nil {
			fs.Destroy()
			t.Fatalf("NewHNSWIndexParams() failed: %v", err)
		}
		err = fs.SetIndexParams(ip)
		ip.Destroy()
		if err != nil {
			fs.Destroy()
			t.Fatalf("SetIndexParams() failed: %v", err)
		}
	}
	if err := cs.AddField(fs); err != nil {
		fs.Destroy()
		t.Fatalf("AddField(%q) failed: %v", name, err)
	}
	fs.Destroy()
}

// addScalarField adds a scalar field to cs.
func addScalarField(t *testing.T, cs *CollectionSchema, name string, dt DataType, nullable bool) {
	t.Helper()
	fs := NewFieldSchema(name, dt, nullable, 0)
	if fs == nil {
		t.Fatalf("NewFieldSchema(%q) returned nil", name)
	}
	if err := cs.AddField(fs); err != nil {
		fs.Destroy()
		t.Fatalf("AddField(%q) failed: %v", name, err)
	}
	fs.Destroy()
}

// newIntrospectSchema builds a schema with two vector fields (one indexed) and
// two scalar fields, exercising every introspection path added for v0.7.0.
func newIntrospectSchema(t *testing.T) *CollectionSchema {
	t.Helper()
	cs := NewCollectionSchema("introspect")
	if cs == nil {
		t.Fatal("NewCollectionSchema() returned nil")
	}
	addVectorField(t, cs, "indexed_vec", 128, true)
	addVectorField(t, cs, "plain_vec", 8, false)
	addScalarField(t, cs, "title", DataTypeString, false)
	addScalarField(t, cs, "score", DataTypeFloat, true)
	return cs
}

func TestCollectionSchemaGetAllFieldNames(t *testing.T) {
	cs := newIntrospectSchema(t)
	defer cs.Destroy()

	names, err := cs.GetAllFieldNames()
	if err != nil {
		t.Fatalf("GetAllFieldNames() failed: %v", err)
	}
	if len(names) != 4 {
		t.Fatalf("GetAllFieldNames() returned %d names (%v), want 4", len(names), names)
	}

	got := make(map[string]bool, len(names))
	for _, n := range names {
		got[n] = true
	}
	for _, want := range []string{"indexed_vec", "plain_vec", "title", "score"} {
		if !got[want] {
			t.Errorf("GetAllFieldNames() missing %q, got %v", want, names)
		}
	}
}

func TestCollectionSchemaGetVectorFields(t *testing.T) {
	cs := newIntrospectSchema(t)
	defer cs.Destroy()

	fields, err := cs.GetVectorFields()
	if err != nil {
		t.Fatalf("GetVectorFields() failed: %v", err)
	}
	if len(fields) != 2 {
		t.Fatalf("GetVectorFields() returned %d fields, want 2", len(fields))
	}

	byName := make(map[string]*FieldSchema, len(fields))
	for _, f := range fields {
		if f == nil {
			t.Fatal("GetVectorFields() returned a nil FieldSchema")
		}
		if !f.IsVectorField() {
			t.Errorf("field %q reported IsVectorField()=false", f.GetName())
		}
		byName[f.GetName()] = f
	}

	indexed := byName["indexed_vec"]
	if indexed == nil {
		t.Fatalf("GetVectorFields() missing indexed_vec, got %v", keys(byName))
	}
	if got := indexed.GetDimension(); got != 128 {
		t.Errorf("indexed_vec GetDimension() = %d, want 128", got)
	}
	if !indexed.HasIndex() {
		t.Error("indexed_vec HasIndex() = false, want true")
	}
	if got := indexed.GetIndexType(); got != IndexTypeHNSW {
		t.Errorf("indexed_vec GetIndexType() = %v, want %v", got, IndexTypeHNSW)
	}

	plain := byName["plain_vec"]
	if plain == nil {
		t.Fatalf("GetVectorFields() missing plain_vec, got %v", keys(byName))
	}
	if got := plain.GetDimension(); got != 8 {
		t.Errorf("plain_vec GetDimension() = %d, want 8", got)
	}
	// zvec assigns a default FLAT/IP index to every vector field on AddField,
	// so a field we never called SetIndexParams on still has one. See
	// TestAddFieldAssignsDefaultVectorIndex.
	if !plain.HasIndex() {
		t.Error("plain_vec HasIndex() = false, want true (zvec default index)")
	}
	if got := plain.GetIndexType(); got != IndexTypeFlat {
		t.Errorf("plain_vec GetIndexType() = %v, want %v (zvec default)", got, IndexTypeFlat)
	}
}

// TestAddFieldAssignsDefaultVectorIndex documents a zvec behavior that is easy
// to get wrong: adding a vector field to a collection schema gives it a
// default FLAT index with metric IP, even when SetIndexParams was never
// called. Scalar fields get no implicit index. Callers introspecting a schema
// must therefore not treat HasIndex() as "the user asked for an index".
func TestAddFieldAssignsDefaultVectorIndex(t *testing.T) {
	cs := NewCollectionSchema("default_index")
	if cs == nil {
		t.Fatal("NewCollectionSchema() returned nil")
	}
	defer cs.Destroy()

	vec := NewFieldSchema("v", DataTypeVectorFP32, false, 8)
	if vec == nil {
		t.Fatal("NewFieldSchema() returned nil")
	}
	// Before AddField the field is standalone and has no index.
	if vec.HasIndex() {
		t.Error("standalone vector field HasIndex() = true, want false")
	}
	if err := cs.AddField(vec); err != nil {
		vec.Destroy()
		t.Fatalf("AddField() failed: %v", err)
	}
	vec.Destroy()

	scalar := NewFieldSchema("s", DataTypeString, false, 0)
	if scalar == nil {
		t.Fatal("NewFieldSchema() returned nil")
	}
	if err := cs.AddField(scalar); err != nil {
		scalar.Destroy()
		t.Fatalf("AddField() failed: %v", err)
	}
	scalar.Destroy()

	gotVec := cs.GetField("v")
	if gotVec == nil {
		t.Fatal("GetField(v) returned nil")
	}
	if !gotVec.HasIndex() {
		t.Error("vector field HasIndex() after AddField = false, want true")
	}
	params := gotVec.GetIndexParams()
	if params == nil {
		t.Fatal("vector field GetIndexParams() after AddField = nil, want default params")
	}
	if got := params.GetType(); got != IndexTypeFlat {
		t.Errorf("default index type = %v, want %v", got, IndexTypeFlat)
	}
	if got := params.GetMetricType(); got != MetricTypeIP {
		t.Errorf("default index metric = %v, want %v", got, MetricTypeIP)
	}

	gotScalar := cs.GetField("s")
	if gotScalar == nil {
		t.Fatal("GetField(s) returned nil")
	}
	if gotScalar.HasIndex() {
		t.Error("scalar field HasIndex() after AddField = true, want false")
	}
	if got := gotScalar.GetIndexParams(); got != nil {
		t.Errorf("scalar field GetIndexParams() = %v, want nil", got)
	}
}

func TestCollectionSchemaGetForwardFields(t *testing.T) {
	cs := newIntrospectSchema(t)
	defer cs.Destroy()

	fields, err := cs.GetForwardFields()
	if err != nil {
		t.Fatalf("GetForwardFields() failed: %v", err)
	}
	if len(fields) != 2 {
		t.Fatalf("GetForwardFields() returned %d fields, want 2", len(fields))
	}

	byName := make(map[string]*FieldSchema, len(fields))
	for _, f := range fields {
		if f.IsVectorField() {
			t.Errorf("GetForwardFields() returned vector field %q", f.GetName())
		}
		byName[f.GetName()] = f
	}

	title := byName["title"]
	if title == nil {
		t.Fatalf("GetForwardFields() missing title, got %v", keys(byName))
	}
	if got := title.GetDataType(); got != DataTypeString {
		t.Errorf("title GetDataType() = %v, want %v", got, DataTypeString)
	}
	if title.IsNullable() {
		t.Error("title IsNullable() = true, want false")
	}

	score := byName["score"]
	if score == nil {
		t.Fatalf("GetForwardFields() missing score, got %v", keys(byName))
	}
	if got := score.GetDataType(); got != DataTypeFloat {
		t.Errorf("score GetDataType() = %v, want %v", got, DataTypeFloat)
	}
	if !score.IsNullable() {
		t.Error("score IsNullable() = false, want true")
	}
}

func TestFieldSchemaGetIndexParams(t *testing.T) {
	cs := newIntrospectSchema(t)
	defer cs.Destroy()

	fields, err := cs.GetVectorFields()
	if err != nil {
		t.Fatalf("GetVectorFields() failed: %v", err)
	}

	var indexed *FieldSchema
	for _, f := range fields {
		if f.GetName() == "indexed_vec" {
			indexed = f
		}
	}
	if indexed == nil {
		t.Fatalf("GetVectorFields() missing indexed_vec, got %v", fieldNames(fields))
	}

	// A field that genuinely has no index must yield nil, not an error or a
	// bogus handle. Scalar fields get no implicit index from AddField, so use
	// one here; vector fields always end up with a default FLAT index.
	scalars, err := cs.GetForwardFields()
	if err != nil {
		t.Fatalf("GetForwardFields() failed: %v", err)
	}
	if len(scalars) == 0 {
		t.Fatal("GetForwardFields() returned no fields")
	}
	if got := scalars[0].GetIndexParams(); got != nil {
		t.Errorf("unindexed scalar GetIndexParams() = %v, want nil", got)
	}

	params := indexed.GetIndexParams()
	if params == nil {
		t.Fatal("indexed_vec GetIndexParams() returned nil, want HNSW params")
	}
	if got := params.GetType(); got != IndexTypeHNSW {
		t.Errorf("GetType() = %v, want %v", got, IndexTypeHNSW)
	}
	if got := params.GetMetricType(); got != MetricTypeCosine {
		t.Errorf("GetMetricType() = %v, want %v", got, MetricTypeCosine)
	}
	if got := params.GetHNSWM(); got != 16 {
		t.Errorf("GetHNSWM() = %d, want 16", got)
	}
	if got := params.GetHNSWEfConstruction(); got != 200 {
		t.Errorf("GetHNSWEfConstruction() = %d, want 200", got)
	}

	// The params are borrowed from the field schema, so Destroy must be a
	// no-op: the field must still report its index afterwards, and a second
	// read must still succeed rather than touching freed memory.
	params.Destroy()
	if !indexed.HasIndex() {
		t.Error("HasIndex() = false after Destroy() on borrowed params, want true")
	}
	again := indexed.GetIndexParams()
	if again == nil {
		t.Fatal("GetIndexParams() returned nil after Destroy() on borrowed params")
	}
	if got := again.GetHNSWM(); got != 16 {
		t.Errorf("GetHNSWM() after borrowed Destroy() = %d, want 16", got)
	}
}

func TestIndexParamsGetIVFParams(t *testing.T) {
	params := NewIndexParams(IndexTypeIVF)
	if params == nil {
		t.Fatal("NewIndexParams(IVF) returned nil")
	}
	defer params.Destroy()

	if err := params.SetMetricType(MetricTypeL2); err != nil {
		t.Fatalf("SetMetricType() failed: %v", err)
	}
	if err := params.SetIVFParams(256, 12, true); err != nil {
		t.Fatalf("SetIVFParams() failed: %v", err)
	}

	nList, nIters, useSoar, err := params.GetIVFParams()
	if err != nil {
		t.Fatalf("GetIVFParams() failed: %v", err)
	}
	if nList != 256 {
		t.Errorf("GetIVFParams() nList = %d, want 256", nList)
	}
	if nIters != 12 {
		t.Errorf("GetIVFParams() nIters = %d, want 12", nIters)
	}
	if !useSoar {
		t.Error("GetIVFParams() useSoar = false, want true")
	}
}

func TestIndexParamsGetInvertParams(t *testing.T) {
	params, err := NewInvertIndexParams(true, false)
	if err != nil {
		t.Fatalf("NewInvertIndexParams() failed: %v", err)
	}
	defer params.Destroy()

	rangeOpt, wildcard, err := params.GetInvertParams()
	if err != nil {
		t.Fatalf("GetInvertParams() failed: %v", err)
	}
	if !rangeOpt {
		t.Error("GetInvertParams() enableRangeOpt = false, want true")
	}
	if wildcard {
		t.Error("GetInvertParams() enableWildcard = true, want false")
	}
}

// TestIndexParamsGetOnNil guards the nil-receiver paths, which must return an
// error rather than dereferencing a nil handle.
func TestIndexParamsGetOnNil(t *testing.T) {
	var params *IndexParams
	if _, _, _, err := params.GetIVFParams(); err == nil {
		t.Error("GetIVFParams() on nil = nil error, want error")
	}
	if _, _, err := params.GetInvertParams(); err == nil {
		t.Error("GetInvertParams() on nil = nil error, want error")
	}
}

// TestCollectionSchemaIntrospectionAfterDestroy checks the borrowed-handle
// guards: once the schema is destroyed, introspection must fail cleanly instead
// of reading freed memory.
func TestCollectionSchemaIntrospectionAfterDestroy(t *testing.T) {
	cs := newIntrospectSchema(t)
	cs.Destroy()

	if _, err := cs.GetAllFieldNames(); err == nil {
		t.Error("GetAllFieldNames() after Destroy() = nil error, want error")
	}
	if _, err := cs.GetVectorFields(); err == nil {
		t.Error("GetVectorFields() after Destroy() = nil error, want error")
	}
	if _, err := cs.GetForwardFields(); err == nil {
		t.Error("GetForwardFields() after Destroy() = nil error, want error")
	}
}

func keys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func fieldNames(fields []*FieldSchema) []string {
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		out = append(out, f.GetName())
	}
	return out
}
