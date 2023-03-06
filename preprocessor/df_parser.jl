module df_parser
    using PyCall
    

    function getdf(path)
        py"""
        import pandas as pd

        def read_csv(path):
            return pd.read_csv(path)
        """
        data = py"read_csv"(path)
        return data
    end

    function dfToMatrix(df)
        data_matrix = Array{Float64}(undef, 0, length(df.columns))

        for i in df.index
            data_matrix = vcat(data_matrix, [convert(Float64, j) for j in df.loc[convert(Int64, i) + 1]]')
        end

        return data_matrix
    end

    function dfToStringMatrix(df)
        data_matrix = Array{String}(undef, 0, length(df.columns))

        for i in df.index
            data_matrix = vcat(data_matrix, reshape([j for j in df.loc[convert(Int64, i) + 1]], 1, length(df.columns)))
        end

        return data_matrix
    end
end